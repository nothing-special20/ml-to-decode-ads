import sys, os
import json
import re
from datetime import datetime

import requests
import numpy as np
import pandas as pd

import torch

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

import nltk
from nltk.corpus import stopwords

from groq import Groq

from sklearn.cluster import MiniBatchKMeans

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_utils import (
    GROQ_API_KEY,
    LI_ADS_LIBRARY_FOLDER,
    DIGITAL_MARKETING_ANALYSIS_FOLDER,
)

li_export_ad_page_file = LI_ADS_LIBRARY_FOLDER + "linkedin_ad_page.json"

li_ad_analysis_file = DIGITAL_MARKETING_ANALYSIS_FOLDER + "li_ad_analysis.json"


def groq_llama_vision(image_url):
    client = Groq(api_key=GROQ_API_KEY)
    prompt = """Analyze this advertisement and return the following in JSON format:
{
   "hook": "Main attention-grabbing message or headline",
   "cta": "Are either call to action text or button listed? if so, what do they say",
   "target_audience": "Intended demographic/audience",
   "image_description": "Brief visual description",
   "text": "All text visible in the image"
}
Note: Use "_" if any element cannot be identified.
Respond only with valid JSON. Do not write an introduction or summary.
"""

    completion = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    return completion.choices[0].message.content


def groq_llama_cluster_analysis(text):
    client = Groq(api_key=GROQ_API_KEY)
    _prompt = "I am giving you a list of hooks. I need you to please analyze them and tell me what they have in common. "
    prompt = _prompt + text

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    return completion.choices[0].message.content


def extract_json(text):
    # Pattern to match balanced braces
    pattern = r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"

    try:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        return None
    except json.JSONDecodeError:
        return None


def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and non-alphabetic tokens
    tokens = [
        word for word in tokens if word.isalpha() and word.lower() not in stop_words
    ]
    # Rejoin the tokens into a single string
    return " ".join(tokens)


def create_embeddings(data, cluster_field, batch_size=32):
    data_for_clusters = data[[cluster_field]]
    data_for_clusters = data_for_clusters[
        [len(x) > 0 for x in data_for_clusters[cluster_field]]
    ]
    data_for_clusters.drop_duplicates(inplace=True)

    cleaned_field_name = f"cleaned_{cluster_field}"
    data_for_clusters[cleaned_field_name] = data_for_clusters[cluster_field].apply(
        preprocess_text
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

    # Process in batches
    embeddings_list = []
    texts = data_for_clusters[cleaned_field_name].tolist()

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.encode(
            batch,
            convert_to_tensor=False,
            show_progress_bar=True,
            batch_size=batch_size,
        )
        embeddings_list.append(batch_embeddings)

    embeddings = np.vstack(embeddings_list)
    return data_for_clusters, embeddings


def create_cluster(data, field_to_cluster_on, num_clusters):
    start_time = datetime.now()
    data_for_clusters, embeddings = create_embeddings(data, field_to_cluster_on)

    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=42,
        batch_size=1024,  # Adjust based on your memory constraints
        init="k-means++",
        max_iter=100,  # Reduced from 300
        n_init=3,  # Reduced from 10
    )

    cluster_field_name = field_to_cluster_on + "_cluster"
    data_for_clusters[cluster_field_name] = kmeans.fit_predict(embeddings)

    print("Time to cluster data using V2 method:\t", datetime.now() - start_time)
    print(f"Total {field_to_cluster_on} count:\t", len(data_for_clusters))
    print(
        "Average time per job title:\t",
        (datetime.now() - start_time) / len(data_for_clusters),
    )

    cluster_data = data_for_clusters[[cluster_field_name, field_to_cluster_on]]

    final_data = pd.merge(cluster_data, data, on=field_to_cluster_on, how="right")

    return final_data


def parse_number(s):
    s = s.strip().lower()
    if ">" in s or "<" in s:
        return 0
    if "k" in s:
        return float(s.replace("k", "")) * 1000
    if "m" in s:
        return float(s.replace("m", "")) * 1000000
    return int(round(float(s), 0))


def parse_percentage(s):
    s = s.replace("\u200a", "").strip()
    if "<" in s:
        return 0

    return float(s.replace("%", "")) / 100


def extract_us_impressions(row):
    try:
        total_str = row["impressions_total"].replace("\u200a", "")

        if "-" in total_str:
            min_val, max_val = map(parse_number, total_str.split("-"))
            total = (min_val + max_val) / 2
        else:
            total = (
                0 if ("<" in total_str or total_str == "_") else parse_number(total_str)
            )

        us_percentage = next(
            (
                parse_percentage(country["percentage"])
                for country in row["impressions_by_country"]
                if country["country"] == "United States"
            ),
            0,
        )

        return total * us_percentage
    except (KeyError, AttributeError, TypeError):
        return None


def extract_company_id(url):
    try:
        return url.split("/company/")[1].split("?")[0]
    except (IndexError, AttributeError):
        return None


def normalize_json(json_string):
    df = pd.read_json(json_string, lines=True)
    df = pd.json_normalize(df.to_dict("records"), sep="_", max_level=2)
    df["us_impressions"] = df.apply(extract_us_impressions, axis=1)
    df["company_id"] = df["basic_info_advertiser_url"].apply(extract_company_id)
    return df


def save_json_data(data_dict, filename, **kwargs):
    try:
        # Open file with append mode
        with open(filename, "a") as file:
            # Add new line and dump json
            json_line = json.dumps(data_dict, **kwargs) + "\n"
            file.write(json_line)
    except Exception as e:
        print("Ugh! Error:", e)


def is_gif_animated(url):
    try:
        response = requests.get(url)
        return (
            response.headers["Content-Type"] == "image/gif"
            and b"NETSCAPE2.0" in response.content
        )
    except:
        return False


if __name__ == "__main__":
    if sys.argv[1] == "analyze_single_image":
        img_url = "https://media.licdn.com/dms/image/v2/D4D10AQFKj01aplFBBw/image-shrink_1280/image-shrink_1280/0/1733761358081/V5_Hustle-TheFutureofNewsletters_MetaPAIDAds2-31png?e=1736964000&v=beta&t=6J7slfn30C4bi8hYuV4llY5PUbRf5i8TGdep4JUEB_A"
        data = extract_json(groq_llama_vision(img_url))
        print(json.dumps(data, indent=4))

    if sys.argv[1] == "analyze_images":
        df = normalize_json(li_export_ad_page_file)
        df = df[~df["image_url"].isna()]
        print(df)

        # CRM data
        company_id_list = [
            "17345",
            "2729320",
            "38373",
            "1876917",
            "2525169",
            "19011410",
        ]
        df = df[df["company_id"].isin(company_id_list)]
        print(df)

        print(li_ad_analysis_file)

        try:
            with open(li_ad_analysis_file, "r") as f:
                analyzed_urls = {
                    json.loads(line).get("ad_url") for line in f if line.strip()
                }
        except (FileNotFoundError, json.JSONDecodeError):
            analyzed_urls = []

        print(analyzed_urls)

        # Or using itertuples() (more efficient):
        for i, row in enumerate(reversed(list(df.itertuples()))):
            ad_url = getattr(row, "url", None)
            if ad_url in analyzed_urls or ad_url is None:
                if ad_url in analyzed_urls:
                    print("already loaded:\t", ad_url)

                continue

            if i >= 500:
                break

            if row.image_url:
                if is_gif_animated(row.image_url):
                    print("url contains a gif - skipping")
                    continue

                analysis = extract_json(groq_llama_vision(row.image_url))
                if analysis:
                    analysis = {
                        "company_id": getattr(row, "company_id", None),
                        "ad_url": ad_url,
                        "us_impressions": getattr(row, "us_impressions", None),
                        "image_url": getattr(row, "image_url", None),
                        **analysis,
                    }
                    print(json.dumps(analysis, indent=4))
                    save_json_data(analysis, li_ad_analysis_file)
                else:
                    print("error with analysis for row.image_url:\t", row.image_url)

        # image_url = "https://media.licdn.com/dms/image/v2/D4D10AQG6In56vtxiBQ/image-shrink_1280/image-shrink_1280/0/1732112713584/Frame48horizontal_v1png?e=2147483647&v=beta&t=eXHDRZz_jGUq8SXRhKQsDTEyNYgr5OB6A6Idb0PY5is"

    if sys.argv[1] == "cluster_image_analysis":
        df = pd.read_json(li_ad_analysis_file, lines=True)
        df = df[~df["ad_url"].isna()]
        del_fields = fields = [
            "call_to_action",
            "ctas",
            "website",
            "ctatext",
            "marketplace",
            "target_audiance",
            "brand",
            "url",
            "sections",
            "logos",
            "call_to_action_text",
            "ctabutton",
            "keywords",
        ]
        for x in del_fields:
            del df[x]
        cluster_field = "hook"
        df[cluster_field] = df[cluster_field].astype(str)
        df = create_cluster(df, cluster_field, 20)

        cluster_cols = [
            cluster_field + "_cluster",
            "company_id",
            "hook",
            "text",
            "ad_url",
            "us_impressions",
            "image_url",
            "call_to_actions",
            "target_audience",
            "image_description",
            "cta",
        ]

        df.to_csv(
            DIGITAL_MARKETING_ANALYSIS_FOLDER + "cluster_image_analysis.csv", index=None
        )

        cluster_data = df.groupby(cluster_field + "_cluster")[cluster_field].agg(lambda x: list(set(x))).reset_index()

        for index, row in cluster_data.iterrows():
            cluster = row[cluster_field + "_cluster"]
            print(cluster)
            # row[cluster_field] is now a list of all hooks for this cluster
            hooks = "\n".join([f"{i+1}) {hook.strip()}" for i, hook in enumerate(row[cluster_field])])
            print(hooks)
            
            analysis = groq_llama_cluster_analysis(hooks)
            
            with open(DIGITAL_MARKETING_ANALYSIS_FOLDER + 'cluster_analysis.md', 'a', encoding='utf-8') as f:
                f.write(f"\nCluster: {cluster}\n")
                f.write(f"Hooks:\n{hooks}\n")
                f.write(f"Analysis: {analysis}\n")
                f.write("-" * 50 + "\n")
