import logging
import re
import spacy
from telegram import Update, BotCommand, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from typing import Dict, Any, Optional, List


from .config import (
    TELEGRAM_TOKEN,
    NER_MODEL_PATH,
    HARDWARE_TEMPLATES,
    identify_hardware_type,
    ALL_CHARACTERISTIC_NAMES
)

from .scraper import scrape_and_process


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


MODEL_TRAINED_FEATURES = []
try:
    nlp_ner = spacy.load(NER_MODEL_PATH)
    if "ner" in nlp_ner.pipe_names:
        MODEL_TRAINED_FEATURES = list(nlp_ner.get_pipe("ner").labels)
    logging.info(f"Модель NER успешно загружена из {NER_MODEL_PATH}")
    logging.info(f"Обученные модели этикеток: {MODEL_TRAINED_FEATURES}")
except OSError:
    logging.error(f"Модель NER не найдена на {NER_MODEL_PATH}. Бот будет возвращаться к правилам, где это возможно, но часть NER работать не будет.")
    nlp_ner = None
except Exception as e:
    logging.error(f"Ошибка загрузки модели NER: {e}", exc_info=True)
    nlp_ner = None


def extract_characteristics_ner(text: str) -> dict[str, list[str]]:

    if nlp_ner is None:
        logging.warning("Модель NER не загружена. Невозможно извлечь с помощью NER.")
        return {}
    if not text or not text.strip():
        logging.warning("Для извлечения NER предоставлен пустой текст.")
        return {}

    extracted_data = {}
    try:
        doc = nlp_ner(text)
        for ent in doc.ents:
            label = ent.label_
            value = ent.text.strip()
            if label not in extracted_data: extracted_data[label] = []
            if value and value not in extracted_data[label]:
                 extracted_data[label].append(value)
        return extracted_data
    except Exception as e:
        logging.error(f"Ошибка в процессе извлечения NER: {e}", exc_info=True)
        return {}


def extract_characteristics_via_rules(clean_text: str, template: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[str]]:

    extracted_data = {key: None for key in template.keys()}
    if not clean_text or not template: return extracted_data
    MAX_KEYWORD_DISTANCE_RULES = 60
    for char_name in sorted(template.keys()):
        rules = template[char_name]
        keywords = rules.get("keywords", [])
        regex_pattern = rules.get("regex")
        if not regex_pattern: continue
        best_match_value = None
        try:

            if keywords:
                potential_matches = list(re.finditer(regex_pattern, clean_text, re.IGNORECASE))
                if potential_matches:
                    keyword_locations = [];
                    for keyword in keywords:
                        if not keyword: continue
                        kw_pattern = re.escape(keyword.lower());
                        for match in re.finditer(kw_pattern, clean_text, re.IGNORECASE): keyword_locations.append(match.span())
                    if keyword_locations:
                        best_candidate_match = None; min_distance_found = float('inf')
                        for value_match in potential_matches:
                            current_min_dist_for_value = float('inf')
                            for kw_start, kw_end in keyword_locations: dist = min(abs(value_match.start() - kw_end), abs(value_match.end() - kw_start)); current_min_dist_for_value = min(current_min_dist_for_value, dist)
                            if current_min_dist_for_value <= MAX_KEYWORD_DISTANCE_RULES and current_min_dist_for_value < min_distance_found:
                                min_distance_found = current_min_dist_for_value; best_candidate_match = value_match
                        if best_candidate_match: value = best_candidate_match.group(1) if best_candidate_match.groups() else best_candidate_match.group(0); best_match_value = value.strip()

            if best_match_value is None:
                 match = re.search(regex_pattern, clean_text, re.IGNORECASE)
                 if match: value = match.group(1) if match.groups() else match.group(0); best_match_value = value.strip()

            if best_match_value:
                 final_value = best_match_value.replace(',', '.').split('(')[0].strip().rstrip('.,;*')
                 extracted_data[char_name] = final_value if final_value else None
        except re.error as e_re: logging.warning(f"Ошибка регулярного выражения правила для '{char_name}': {e_re}")
        except Exception as e: logging.error(f"Ошибка при извлечении правила для '{char_name}': {e}", exc_info=False)
    return extracted_data


def start(update: Update, context: CallbackContext) -> None:

    user = update.effective_user
    update.message.reply_html(
        rf"Привет, {user.mention_html()}! Я помогу извлечь технические характеристики."
        "\nОтправь мне сообщение в формате:\n"
        "<code>Название Оборудования</code>\n"
        "<code>https://ссылка_на_страницу</code>",
        reply_to_message_id=update.message.message_id
    )

def help_command(update: Update, context: CallbackContext) -> None:

    if MODEL_TRAINED_FEATURES: labels_str = ", ".join(sorted(MODEL_TRAINED_FEATURES))
    else: labels_str = "(Список недоступен, NER модель не загружена)"
    supported_types = ", ".join(HARDWARE_TEMPLATES.keys())

    help_text = (
        "Отправь сообщение:\n```\nНазвание Оборудования\nURL\n```\n"
        "Я извлеку характеристики гибридным методом:\n"
        "1. **NER модель** определяет тип характеристики.\n"
        "2. **Парсер правил** предоставляет точное значение (приоритет).\n"
        "3. Если парсер не нашел, используется текст от NER (fallback).\n\n"
        f"*Метки NER:*\n_{labels_str}_\n\n"
        f"*Типы для правил:*\n_{supported_types}_"
         )
    update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


def extract_info_from_message(text: str) -> tuple[str | None, str | None]:

    lines = text.strip().split('\n'); name, url = (None, None)
    if len(lines) >= 1: name = lines[0].strip()
    url_pattern = r'https?://[^\s]+'
    for i in range(len(lines)):
        line = lines[i].strip()
        if line: match = re.search(url_pattern, line)
        if match: url = match.group(0); break
    return (name, url) if name and url else (None, None)



def handle_message(update: Update, context: CallbackContext) -> None:

    message_text = update.message.text
    name, url = extract_info_from_message(message_text)
    if not url or not name: update.message.reply_text("Неверный формат. Нужны название и URL.", reply_to_message_id=update.message.message_id); return

    processing_msg = update.message.reply_text(f"Анализирую '{name}'...", reply_to_message_id=update.message.message_id)
    logger.info(f"Обработка запроса: Имя='{name}', Ссылка='{url}'")


    try:
        _, clean_text, _ = scrape_and_process(url)
        if clean_text is None: context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=processing_msg.message_id, text=f"Ошибка загрузки/извлечения текста: {url}"); return
        if not clean_text.strip(): context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=processing_msg.message_id, text=f"Нет текста для анализа: {url}"); return
    except Exception as e_scrape: logger.error(f"Ошибка крапинга для {url}: {e_scrape}", exc_info=True); context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=processing_msg.message_id, text=f"Ошибка обработки страницы {url}."); return

    logger.info(f"Получен чистый текст (длина {len(clean_text)})...")


    extracted_chars_rules_raw = {}
    rule_extraction_error = None


    if nlp_ner:
        logger.info("Запуск модели NER...")
        extracted_chars_ner = extract_characteristics_ner(clean_text)
        logger.info(f"Необработанный результат NER: {extracted_chars_ner}")
    else: logger.warning("Модель NER не загружена, извлечение NER пропускается.")


    hw_type = identify_hardware_type(name); logger.info(f"Определенный тип правил: {hw_type}")
    if hw_type and hw_type in HARDWARE_TEMPLATES:
        template = HARDWARE_TEMPLATES[hw_type]
        try:
            logger.info("Запуск парсера на основе правил...")
            extracted_chars_rules_raw = extract_characteristics_via_rules(clean_text, template)
            logger.info(f"Исходный результат на основе правил: {extracted_chars_rules_raw}")
        except Exception as e_extract_rules:
            logger.error(f"Ошибка извлечения правила: {e_extract_rules}", exc_info=True); rule_extraction_error = str(e_extract_rules)
    elif not hw_type: logger.warning(f"Невозможно запустить анализатор правил: Неизвестный тип для '{name}'.")
    else: logger.warning(f"Невозможно запустить анализатор правил: нет шаблона для типа '{hw_type}'.")


    final_results = {}
    processed_ner_labels = set()


    if extracted_chars_ner:
        for label, ner_texts in extracted_chars_ner.items():
            processed_ner_labels.add(label)

            rule_value = extracted_chars_rules_raw.get(label)


            if rule_value is not None and rule_value != "":

                final_results[label] = rule_value
                logging.debug(f"Объединить '{label}': Использование значения правила '{rule_value}' (НЭР найден: {ner_texts})")
            elif ner_texts:

                final_results[label] = ner_texts[0]
                logging.debug(f"Объединить '{label}': Использование значения NER '{ner_texts[0]}' (Обнаружен парсер правил. Нет/пусто)")



    if extracted_chars_rules_raw:
        for label, rule_value in extracted_chars_rules_raw.items():

            if label not in processed_ner_labels and rule_value is not None and rule_value != "":
                final_results[label] = rule_value
                logging.debug(f"Объединить '{label}': Добавление значения, найденного ТОЛЬКО анализатором правил: '{rule_value}'")

    logger.info(f"Окончательный совокупный результат: {final_results}")

    reply_parts = [f"Характеристики для **{name}** (NER+Парсер):"]
    if not final_results:
        reply_parts.append("\n_(Ничего не найдено)_")
        if nlp_ner is None: reply_parts.append("\n_(NER модель не загружена)_")
        if rule_extraction_error: reply_parts.append(f"\n_(Ошибка парсера правил: {rule_extraction_error})_")
    else:
        for char_name in sorted(final_results.keys()):
            value = final_results[char_name]
            reply_parts.append(f"  • _{char_name}_: <code>{value}</code>")

    reply_text = "\n".join(reply_parts)

    try:
        context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=processing_msg.message_id, text=reply_text, parse_mode=ParseMode.HTML)
        logger.info(f"Отправлено объединенных результатов для URL: {url}")
    except Exception as e_send:
         logger.error(f"Не удалось отправить/редактировать сообщение: {e_send}")
         try: update.message.reply_text(reply_text, parse_mode=ParseMode.HTML, reply_to_message_id=update.message.message_id)
         except Exception as e_send_new: logger.error(f"Не удалось отправить новое сообщение: {e_send_new}")

def main() -> None:

    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "ВАШ_ТОКЕН_ЗДЕСЬ": logger.critical("ОШИБКА: Токен Telegram не найден!"); return
    if nlp_ner is None: logger.warning("Модель NER не загружена. Часть NER гибридного подхода будет пропущена.")

    updater = Updater(TELEGRAM_TOKEN); dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("Старт", start)); dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    try: updater.bot.set_my_commands([BotCommand("Старт", "Старт"), BotCommand("Помощь", "Помощь")]); logger.info("Набор команд.")
    except Exception as e_cmd: logger.warning(f"Не удалось выполнить команду: {e_cmd}")
    updater.start_polling(); logger.info("Бот начал опрос..."); updater.idle(); logger.info("Бот остановлен.")


if __name__ == '__main__':
    main()
