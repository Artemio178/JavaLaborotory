import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random
import logging
import re
import json
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional


from .config import (
    DATASET_PATH,
    NER_MODEL_PATH,
    ALL_CHARACTERISTIC_NAMES,
    HARDWARE_TEMPLATES,
    identify_hardware_type
)
from .scraper import scrape_and_process


LOGGING_LEVEL = logging.DEBUG
logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')




MAX_KEYWORD_DISTANCE = 60
ALIGNMENT_MODE = "strict"
LOG_LIMIT_PER_URL = 15


def extract_spans_via_rules(clean_text: str, template: Dict[str, Dict[str, Any]]) -> List[Tuple[int, int, str]]:
    extracted_spans = []
    if not clean_text or not template:
        return extracted_spans
    occupied_spans = set()


    for char_name in sorted(template.keys()):
        rules = template[char_name]
        keywords = rules.get("keywords", [])
        regex_pattern = rules.get("regex")


        if not regex_pattern:
            continue

        best_match_span = None
        try:

            if keywords:
                potential_matches = list(re.finditer(regex_pattern, clean_text, re.IGNORECASE))
                if potential_matches:
                    keyword_locations = []
                    for keyword in keywords:
                        if not keyword: continue
                        kw_pattern = re.escape(keyword.lower())
                        for match in re.finditer(kw_pattern, clean_text, re.IGNORECASE):
                            keyword_locations.append(match.span())


                    if keyword_locations:
                        best_candidate_match = None
                        min_distance_found = float('inf')


                        for value_match in potential_matches:
                            current_min_dist_for_value = float('inf')
                            for kw_start, kw_end in keyword_locations:

                                dist = min(abs(value_match.start() - kw_end), abs(value_match.end() - kw_start))
                                current_min_dist_for_value = min(current_min_dist_for_value, dist)


                            if current_min_dist_for_value <= MAX_KEYWORD_DISTANCE and current_min_dist_for_value < min_distance_found:
                                current_span = value_match.span()

                                if current_span[0] == current_span[1]: continue


                                is_overlapping = False
                                for occ_start, occ_end in occupied_spans:
                                    if max(current_span[0], occ_start) < min(current_span[1], occ_end):
                                        is_overlapping = True; break

                                if not is_overlapping:
                                    min_distance_found = current_min_dist_for_value
                                    best_candidate_match = value_match


                        if best_candidate_match:
                            best_match_span = best_candidate_match.span()


            if best_match_span is None:

                 match = re.search(regex_pattern, clean_text, re.IGNORECASE)
                 if match:
                     current_span = match.span()

                     if current_span[0] < current_span[1]:

                          is_overlapping = False
                          for occ_start, occ_end in occupied_spans:
                               if max(current_span[0], occ_start) < min(current_span[1], occ_end):
                                    is_overlapping = True; break
                          if not is_overlapping:
                               best_match_span = current_span


            if best_match_span:
                 if best_match_span[0] < best_match_span[1]:
                     extracted_spans.append((best_match_span[0], best_match_span[1], char_name))
                     occupied_spans.add(best_match_span)

        except re.error as e_re:
            logging.warning(f"Regex error during span extraction for '{char_name}': {e_re}")
        except Exception as e:

            logging.error(f"Unexpected error during rule-based span extraction for '{char_name}': {e}", exc_info=False)


    extracted_spans.sort(key=lambda x: x[0])
    return extracted_spans


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip().lower()


def train_ner_model(input_csv_path: str, model_save_path: str, n_iter: int = 75, base_model: str = "ru_core_news_sm"):
    try:
        df_input = pd.read_csv(input_csv_path, encoding='utf-8');
        if 'name' not in df_input.columns or 'url' not in df_input.columns: logging.error(f"Input needs 'name','url'"); return
        df_input = df_input.dropna(subset=['url']); df_input = df_input[df_input['url'].str.strip() != '']; df_input = df_input.drop_duplicates(subset=['url'])
        logging.info(f"Loaded {len(df_input)} unique valid URLs from {input_csv_path}")
        if df_input.empty: logging.error("No valid URLs."); return
    except FileNotFoundError: logging.error(f"Not found: {input_csv_path}."); return
    except Exception as e_read: logging.error(f"Error reading {input_csv_path}: {e_read}", exc_info=True); return


    training_data = []
    processed_urls, skipped_scraping, skipped_no_text, skipped_no_type = 0, 0, 0, 0
    pages_with_no_entities, total_spans_generated = 0, 0

    logging.info("Generating training data using scraper and rule-based parser...")
    for index, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Generating NER data"):
        url = row['url']; name = row.get('name', ''); processed_urls += 1
        _, clean_text, _ = scrape_and_process(url)
        if clean_text is None: skipped_scraping += 1; logging.debug(f"Skip URL {index} (Scraping fail): {url}"); continue
        if not clean_text.strip(): skipped_no_text += 1; logging.debug(f"Skip URL {index} (No text): {url}"); continue
        hw_type = identify_hardware_type(name) if name else None
        if not hw_type or hw_type not in HARDWARE_TEMPLATES: skipped_no_type += 1; logging.debug(f"Skip URL {index} (No type/template: '{hw_type}'): {url}"); continue
        template = HARDWARE_TEMPLATES[hw_type]

        logging.debug(f"\n--- Processing URL {index}: {url} (Type: {hw_type}) ---")
        entities = extract_spans_via_rules(clean_text, template)

        if entities:

            logging.info(f"Rule Parser found {len(entities)} entities for URL {index} (showing Label: Value):")

            for i, (start, end, label) in enumerate(entities):

                entity_text = clean_text[start:end].strip()

                log_entry = f"  - {label}: '{entity_text}'"

                logging.info(log_entry)
                if i == LOG_LIMIT_PER_URL:
                     logging.info(f"  ... (further {len(entities) - LOG_LIMIT_PER_URL} entities hidden)")
                     break



            training_data.append({'text': clean_text, 'entities': entities})
            total_spans_generated += len(entities)
        else:
            logging.debug(f"Rule Parser found 0 entities for URL {index}.")
            pages_with_no_entities += 1
        logging.debug(f"--- End Processing URL {index} ---")

    if not training_data:
        logging.error("Stopping training: No training examples were generated by the rule parser.")
        return


    try:
        nlp = spacy.load(base_model)
        logging.info(f"Loaded pre-trained model '{base_model}' as base.")
    except OSError:
        logging.error(f"Base model '{base_model}' not found. Please download it: python -m spacy download {base_model}")
        logging.info("Using blank 'ru' model instead.")
        nlp = spacy.blank("ru")


    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
        logging.info("Added 'ner' pipe to the model.")
    else:
        ner = nlp.get_pipe("ner")
        logging.info("Found existing 'ner' pipe in the model.")


    characteristics_to_extract = ALL_CHARACTERISTIC_NAMES
    existing_labels = ner.labels
    new_labels_added = 0
    for label in characteristics_to_extract:
        if label not in existing_labels:
            ner.add_label(label)
            new_labels_added += 1
    logging.info(f"Added {new_labels_added} new labels to NER component. Total labels: {len(ner.labels)}")


    db = DocBin()

    skipped_entities_count = 0
    processed_docs_for_db = 0
    spans_discarded_mismatch_or_none = 0
    docs_discarded_fully = 0

    logging.info(f"Creating DocBin with strict span text check (alignment_mode='{ALIGNMENT_MODE}')")

    for item in tqdm(training_data, desc="Creating DocBin (Strict Check)"):
        text = item['text']
        original_entities = item['entities']
        if not text or not original_entities: continue

        try:

            doc = nlp.make_doc(text)
            valid_ents_for_doc = []


            for start, end, label in original_entities:

                if not (0 <= start < end <= len(text)):
                     logging.warning(f"Invalid indices ({start},{end}) for label '{label}'. Skipping entity in doc starting with: '{text[:50]}...'"); skipped_entities_count += 1; continue


                expected_text = text[start:end]
                expected_text_norm = normalize_text(expected_text)

                if not expected_text_norm:
                     logging.warning(f"Empty expected text for label '{label}' at indices ({start},{end}). Skipping entity."); skipped_entities_count += 1; continue


                span = doc.char_span(start, end, label=label, alignment_mode=ALIGNMENT_MODE)


                valid_span_found = False
                if span is not None:

                    span_text_norm = normalize_text(span.text)
                    if span_text_norm == expected_text_norm:
                        valid_span_found = True
                    else:

                        logging.debug(f"Span text mismatch for '{label}': Expected~'{expected_text_norm}', Got~'{span_text_norm}'. DISCARDING.")
                        spans_discarded_mismatch_or_none += 1
                else:

                    logging.debug(f"Cannot create span for '{expected_text}' ({start},{end}, {label}) with mode '{ALIGNMENT_MODE}'. DISCARDING.")
                    spans_discarded_mismatch_or_none += 1


                if valid_span_found:

                    valid_ents_for_doc.append(span)
                else:

                    skipped_entities_count += 1



            if valid_ents_for_doc:
                try:

                    doc.ents = valid_ents_for_doc

                    db.add(doc)
                    processed_docs_for_db += 1
                except ValueError as e_ents:

                    logging.warning(f"Error setting ents (likely overlap after strict check): {e_ents}. Skipping doc: '{text[:50]}...'")
                    docs_discarded_fully += 1
            else:

                docs_discarded_fully += 1
                logging.debug(f"Discarding doc (no valid entities remained after checks): '{text[:50]}...'")

        except Exception as e_doc:

             logging.error(f"Unexpected error processing doc: {e_doc}", exc_info=True)


    if processed_docs_for_db == 0:
        logging.error("DocBin is empty! Cannot train. Check logs for reasons why entities/docs were discarded.")
        return
    logging.info(f"Created DocBin with {processed_docs_for_db} documents.")
    if skipped_entities_count > 0:
        logging.warning(f"Total invalid/skipped entities during DocBin creation: {skipped_entities_count}")
    if spans_discarded_mismatch_or_none > 0:
        logging.warning(f"-> {spans_discarded_mismatch_or_none} entities were discarded due to 'None' span or text mismatch (strict check).")
    if docs_discarded_fully > 0:
        logging.warning(f"-> {docs_discarded_fully} documents were fully discarded (no valid entities remained).")

    db_path = "models/train_data_strict.spacy"
    db.to_disk(db_path)
    logging.info(f"Training data saved to {db_path}")


    from spacy.util import minibatch, compounding
    try:

        db_train = DocBin().from_disk(db_path)

        train_docs = list(db_train.get_docs(nlp.vocab))
        if not train_docs:
            logging.error(f"No documents loaded from {db_path}. Cannot train.")
            return


        train_examples = []
        for doc in train_docs:
            try:

                entities_list = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
                example = Example.from_dict(doc, {"entities": entities_list})
                train_examples.append(example)
            except Exception as e_example:
                logging.warning(f"Failed to create Example object from doc: {e_example}")

        if not train_examples:
            logging.error("No valid Example objects were created for training.")
            return
        logging.info(f"Prepared {len(train_examples)} training examples for spaCy.")

    except FileNotFoundError:
        logging.error(f"Training data file not found: {db_path}. Run data generation steps first."); return
    except Exception as e_load_db:
        logging.error(f"Error loading/processing training data from {db_path}: {e_load_db}", exc_info=True); return


    pipe_exceptions = ["ner"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    logging.info(f"Starting training loop ({n_iter} iterations)...")

    optimizer = nlp.initialize(lambda: train_examples)

    batch_sizes = compounding(4.0, 32.0, 1.001)


    with nlp.disable_pipes(*other_pipes):
        for iteration in range(n_iter):
            random.shuffle(train_examples)
            losses = {}
            current_batch_size = int(next(batch_sizes))

            batches = minibatch(train_examples, size=current_batch_size)
            for batch in tqdm(batches, desc=f"Iter {iteration+1}/{n_iter} (batch ~{current_batch_size})"):
                try:
                     if not batch: continue

                     nlp.update(batch, drop=0.3, losses=losses, sgd=optimizer)
                except Exception as e_update:

                     logging.error(f"Error during nlp.update: {e_update}", exc_info=False)


            ner_loss = losses.get('ner', 0.0)
            logging.info(f"Iter {iteration+1}/{n_iter}, Losses: NER={ner_loss:.4f}")


    try:
        nlp.to_disk(model_save_path)
        logging.info(f"NER model saved successfully to {model_save_path}")
    except Exception as e_save:
        logging.error(f"Error saving NER model to {model_save_path}: {e_save}", exc_info=True)

if __name__ == "__main__":
    logging.info("\n--- Starting NER Model Training (Rule-Based Annotation, Fine-tuning, Strict Span Check) ---")
    train_ner_model(
        input_csv_path=DATASET_PATH,
        model_save_path=NER_MODEL_PATH,
        n_iter=75,
        base_model="ru_core_news_sm"
    )
    logging.info("--- NER Model Training Finished ---")