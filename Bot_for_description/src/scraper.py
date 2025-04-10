import undetected_chromedriver as uc
import logging
import time
import random
import atexit
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import pymorphy2
import re
from selenium.common.exceptions import TimeoutException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


driver = None


try:
    russian_stopwords = stopwords.words("russian")
except LookupError:
    logging.info("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    russian_stopwords = stopwords.words("russian")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logging.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')


try:
    morph = pymorphy2.MorphAnalyzer()
    logging.info("pymorphy2 analyzer initialized.")
except Exception as e:
    logging.error(f"Failed to initialize pymorphy2: {e}", exc_info=True)
    morph = None


def setup_driver():
    global driver
    if driver is None:
        try:
            options = uc.ChromeOptions()
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--start-maximized")
            logging.info("Setting up undetected_chromedriver...")
            driver = uc.Chrome(options=options)
            logging.info("WebDriver setup complete.")
            return driver
        except Exception as e:
            logging.error(f"Failed to setup WebDriver: {e}", exc_info=True)
            logging.error("Please ensure Chrome is installed and 'chromedriver' is available/compatible or path is specified.")
            driver = None
            return None
    return driver

def shutdown_driver():
    global driver
    if driver:
        try:
            driver.quit()
            logging.info("WebDriver shut down successfully.")
        except Exception as e:
            logging.error(f"Error shutting down WebDriver: {e}")
        driver = None


atexit.register(shutdown_driver)


setup_driver()



def fetch_html(url: str, timeout: int = 45) -> str | None:
    global driver
    if driver is None:
        logging.error("WebDriver is not initialized. Attempting to restart.")
        driver = setup_driver()
        if driver is None:
             logging.error("WebDriver restart failed. Cannot fetch HTML.")
             return None

    try:
        logging.info(f"Navigating to {url} using WebDriver...")

        driver.set_page_load_timeout(timeout)
        driver.get(url)

        sleep_duration = random.uniform(3.5, 7.0)
        logging.info(f"Waiting for {sleep_duration:.2f} seconds...")
        time.sleep(sleep_duration)

        html = driver.page_source
        logging.info(f"Successfully fetched HTML from {url} using WebDriver (length: {len(html)})")
        return html
    except TimeoutException:
        logging.error(f"Page load timeout ({timeout}s) occurred for {url}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while fetching {url} with WebDriver: {e}", exc_info=True)
        return None


def extract_text_from_html(html_content: str) -> str:
    if not html_content:
        return ""
    try:

        soup = BeautifulSoup(html_content, 'lxml')


        tags_to_remove = ["script", "style", "head", "meta", "link", "noscript", "iframe", "svg"]
        for tag_name in tags_to_remove:
            for element in soup.find_all(tag_name):
                element.decompose()


        target_node = soup.find('body') or soup
        if target_node:
            text = target_node.get_text(separator='\n', strip=False)
        else:
            text = ""

        lines = []
        for line in text.splitlines():
            cleaned_line = re.sub(r'\s+', ' ', line).strip()
            if cleaned_line:
                lines.append(cleaned_line)


        cleaned_text = "\n\n".join(lines)

        logging.debug(f"Extracted text length (simple method): {len(cleaned_text)}")
        return cleaned_text

    except Exception as e:
        logging.error(f"Error parsing HTML (simple method): {e}", exc_info=True)
        return ""

def preprocess_text(text: str) -> tuple[str, list[str]]:
    lemmas = []
    clean_text_for_rules = ""
    if not text:
        return clean_text_for_rules, lemmas


    clean_text_for_rules = text.lower()

    clean_text_for_rules = re.sub(r'[^\w\s\.,хx/\-+\n%°"\'()]', ' ', clean_text_for_rules, flags=re.IGNORECASE | re.UNICODE)
    clean_text_for_rules = re.sub(r'\s{2,}', ' ', clean_text_for_rules).strip()
    clean_text_for_rules = re.sub(r'\n\s+', '\n', clean_text_for_rules)
    clean_text_for_rules = re.sub(r'\s+\n', '\n', clean_text_for_rules)
    clean_text_for_rules = re.sub(r'\n{3,}', '\n\n', clean_text_for_rules)


    if morph is None:
        logging.warning("pymorphy2 analyzer not available. Skipping lemmatization.")
        return clean_text_for_rules, []

    text_for_lemmas = clean_text_for_rules.replace('\n', ' ')
    text_for_lemmas = re.sub(r'\s{2,}', ' ', text_for_lemmas).strip()

    try:
        tokens = nltk.word_tokenize(text_for_lemmas, language="russian")
        for token in tokens:

            if token.isalnum() and token not in russian_stopwords and len(token) > 1:

                 try:
                     p = morph.parse(token)[0]
                     lemmas.append(p.normal_form)
                 except Exception as e_lemma:

                     lemmas.append(token)

    except Exception as e_tok:
         logging.error(f"Tokenization error: {e_tok}", exc_info=True)


    logging.debug(f"Preprocessing done. Clean text length: {len(clean_text_for_rules)}, Lemmas count: {len(lemmas)}")
    return clean_text_for_rules, lemmas


def scrape_and_process(url: str) -> tuple[str | None, str | None, list[str] | None]:
    logging.info(f"Starting processing for URL: {url}")
    html_content = fetch_html(url)
    if html_content is None:
        logging.warning(f"Failed to fetch HTML for {url}. Returning None.")
        return None, None, None


    raw_text = extract_text_from_html(html_content)
    if not raw_text:
        logging.warning(f"No significant text extracted from {url}.")

        clean_text_for_rules, lemmas = preprocess_text("")
        return html_content, clean_text_for_rules, lemmas


    clean_text_for_rules, lemmas = preprocess_text(raw_text)


    return html_content, clean_text_for_rules, lemmas



