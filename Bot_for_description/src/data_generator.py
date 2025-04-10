import pandas as pd
import logging
import re
import time
import os
import random
from typing import Dict, Any, Optional



from .scraper import scrape_and_process
from .config import (
    HARDWARE_TEMPLATES,
    identify_hardware_type,
    ALL_CHARACTERISTIC_NAMES,
    DATASET_PATH,
    LABELED_DATA_PATH

)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_characteristics_text_based(clean_text: str, template: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[str]]:

    extracted_data = {key: None for key in template.keys()}
    if not clean_text:
        return extracted_data


    for char_name, rules in template.items():
        keywords = rules.get("keywords", [])
        regex_pattern = rules.get("regex")


        if not regex_pattern and not keywords:
            logging.debug(f"No keywords or regex for '{char_name}'. Skipping text extraction.")
            continue

        best_match_value = None

        try:

            if regex_pattern and keywords:

                potential_values = list(re.finditer(regex_pattern, clean_text, re.IGNORECASE))

                if potential_values:
                    best_found_value = None
                    min_distance = 100


                    keyword_locations = []
                    for keyword in keywords:

                        kw_pattern = r'\b' + re.escape(keyword.lower()) + r'\b' if keyword.isalnum() else re.escape(keyword.lower())
                        for match in re.finditer(kw_pattern, clean_text, re.IGNORECASE):
                            keyword_locations.append(match)

                    if keyword_locations:

                         for value_match in potential_values:
                             current_min_dist_for_value = float('inf')

                             for kw_match in keyword_locations:

                                 dist = min(abs(value_match.start() - kw_match.end()), abs(kw_match.start() - value_match.end()))
                                 current_min_dist_for_value = min(current_min_dist_for_value, dist)


                             if current_min_dist_for_value < min_distance:
                                 min_distance = current_min_dist_for_value
                                 value = value_match.group(1) if value_match.groups() else value_match.group(0)
                                 best_found_value = value.strip()


                    if best_found_value:
                        final_value = best_found_value.replace(',', '.')
                        final_value = final_value.split('(')[0].strip().rstrip('.,;*')
                        best_match_value = final_value
                        logging.debug(f"Text match (regex+keyword proximity) for '{char_name}': '{final_value}'")



            if best_match_value is None and regex_pattern:

                 match = re.search(regex_pattern, clean_text, re.IGNORECASE)
                 if match:
                      value = match.group(1) if match.groups() else match.group(0)
                      value = value.strip()
                      final_value = value.replace(',', '.')
                      final_value = final_value.split('(')[0].strip().rstrip('.,;*')
                      best_match_value = final_value
                      logging.debug(f"Text match (regex only) for '{char_name}': '{final_value}'")



            extracted_data[char_name] = best_match_value

        except Exception as e:
            logging.error(f"Error during text-based extraction for '{char_name}': {e}", exc_info=False)

    return extracted_data



def generate_labeled_data(input_csv_path: str, output_csv_path: str):



    try:
        if os.path.exists(output_csv_path):
            os.remove(output_csv_path)
            logging.info(f"Removed existing file: {output_csv_path}")
    except OSError as e:
        logging.error(f"Error removing file {output_csv_path}: {e}. Proceeding anyway.")
    except Exception as e_rem:
         logging.error(f"Unexpected error removing file {output_csv_path}: {e_rem}. Proceeding anyway.")



    try:
        df_input = pd.read_csv(input_csv_path, encoding='utf-8')
        logging.info(f"Read {len(df_input)} rows from {input_csv_path}")

        if 'name' not in df_input.columns or 'url' not in df_input.columns:
            logging.error(f"Input file {input_csv_path} must contain 'name' and 'url' columns.")
            return

    except FileNotFoundError:
        logging.error(f"Input file not found: {input_csv_path}. Please create it with 'name' and 'url' columns.")
        return
    except pd.errors.EmptyDataError:
         logging.error(f"Input file {input_csv_path} is empty.")
         return
    except Exception as e_read:
         logging.error(f"Error reading {input_csv_path}: {e_read}", exc_info=True)
         return



    results = []
    processed_urls = 0
    total_urls = len(df_input)
    all_char_names = ALL_CHARACTERISTIC_NAMES

    for index, row in df_input.iterrows():
        url = row.get('url')
        name = row.get('name')

        if pd.isna(url) or not str(url).strip() or pd.isna(name) or not str(name).strip():
             logging.warning(f"Skipping row {index + 1} due to missing or empty name/URL.")
             continue

        url = str(url).strip()
        name = str(name).strip()

        logging.info(f"Processing row {index + 1}/{total_urls}: {name} ({url})")
        time.sleep(random.uniform(2.0, 4.0))


        html_content, clean_text, lemmas = scrape_and_process(url)


        if html_content is None:
            logging.warning(f"Scraping failed for URL: {url}")
            data_row = {'name': name, 'url': url, 'type': None, 'clean_text': None, 'lemmas': None}
            for char_name in all_char_names: data_row[char_name] = None
            results.append(data_row)
            continue


        hw_type = identify_hardware_type(name)
        logging.info(f"Identified type: {hw_type}")


        extracted_chars = {}

        if hw_type and hw_type in HARDWARE_TEMPLATES and clean_text:
            template = HARDWARE_TEMPLATES[hw_type]
            extracted_chars = extract_characteristics_text_based(clean_text, template)
            logging.info(f"Extracted via text: { {k:v for k,v in extracted_chars.items() if v is not None} }")
        elif not clean_text:
            logging.warning(f"No text could be extracted for {url}. Cannot extract characteristics.")
        else:
            logging.warning(f"Could not determine type or no template for '{name}'. No characteristics extracted.")



        data_row = {
            'name': name, 'url': url, 'type': hw_type,
            'clean_text': clean_text,
            'lemmas': " ".join(lemmas) if lemmas else ""
        }
        for char_name in all_char_names:
            data_row[char_name] = extracted_chars.get(char_name, None)

        results.append(data_row)
        processed_urls += 1


    if results:
        try:
            df_output = pd.DataFrame(results)
            cols_order = ['name', 'url', 'type', 'clean_text', 'lemmas'] + all_char_names
            for col in cols_order:
                if col not in df_output.columns:
                    df_output[col] = None
            df_output = df_output[cols_order]

            df_output.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            logging.info(f"Successfully processed {processed_urls} URLs where HTML was fetched.")
            logging.info(f"Saved labeled data ({len(df_output)} rows) to {output_csv_path}")
        except Exception as e_save:
            logging.error(f"Error saving data to {output_csv_path}: {e_save}", exc_info=True)
    else:
        logging.warning("No data processed successfully. Output file was not created/overwritten.")



if __name__ == "__main__":
    logging.info("--- Starting Data Labeling (Step 2) ---")
    generate_labeled_data(DATASET_PATH, LABELED_DATA_PATH)
    logging.info("--- Data Labeling Finished ---")