import re
from difflib import SequenceMatcher

from prompts import build_messages

def count_input_tokens(product: dict, tokenizer) -> int:
    """Вычисляет количество токенов полного входа."""
    messages = build_messages(product)

    return len(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )


def count_product_tokens_without_reviews(product: dict, tokenizer) -> int:
    """Вычисляет количество токенов карточки без отзывов."""
    product_without_reviews = product.copy()
    product_without_reviews["reviews"] = []

    return count_input_tokens(product_without_reviews, tokenizer)


def normalize_review(review: str) -> str:
    """Нормализует текст отзыва для сравнения."""
    review = review.lower()
    review = re.sub(r"[^\w\s]", " ", review)
    review = re.sub(r"\s+", " ", review)
    return review.strip()


def remove_exact_duplicates(reviews: list[str]) -> list[str]:
    """Удаляет точные дубликаты, сохраняя порядок отзывов."""
    unique_reviews = []
    seen = set()

    for review in reviews:
        normalized_review = normalize_review(review)

        if normalized_review not in seen:
            seen.add(normalized_review)
            unique_reviews.append(review)

    return unique_reviews


def remove_similar_reviews(
        reviews: list[str],
        similarity_threshold: float = 0.9,
) -> list[str]:
    """Удаляет вложенные и сильно похожие отзывы."""
    reviews = sorted(reviews, key=len, reverse=True)
    unique_reviews = []

    for review in reviews:
        normalized_review = normalize_review(review)

        for saved_review in unique_reviews:
            normalized_saved = normalize_review(saved_review)

            if normalized_review in normalized_saved:
                break

            similarity = SequenceMatcher(
                None,
                normalized_review,
                normalized_saved,
            ).ratio()

            if similarity >= similarity_threshold:
                break
        else:
            unique_reviews.append(review)

    return unique_reviews


def fit_reviews_to_token_limit(
        product: dict,
        reviews: list[str],
        tokenizer,
        max_input_tokens: int = 1500
) -> list[str]:
    """Удаляет последние отзывы, пока полный вход не уложится в лимит."""
    reviews = reviews.copy()

    while reviews:
        shortened_product = product.copy()
        shortened_product["reviews"] = reviews

        if count_input_tokens(shortened_product, tokenizer) <= max_input_tokens:
            break

        reviews.pop()

    return reviews


def selection_of_reviews(review_budget: int, product: dict, tokenizer, max_input_tokens: int = 1500) -> list[str]:
    """Поочередно отбирает длинные и короткие отзывы."""
    reviews = sorted(product["reviews"], key=len)

    selected = []
    left = 0
    right = len(reviews) - 1

    while left <= right:
        review = reviews[right]
        num_tokens = len(tokenizer.encode(review))

        if num_tokens <= review_budget:
            selected.append(review)
            review_budget -= num_tokens

        right -= 1

        if left > right:
            break

        review = reviews[left]
        num_tokens = len(tokenizer.encode(review))

        if num_tokens <= review_budget:
            selected.append(review)
            review_budget -= num_tokens

        left += 1

    return fit_reviews_to_token_limit(product, selected, tokenizer, max_input_tokens,)


def prepare_product_input(
        product: dict,
        tokenizer,
        max_input_tokens: int = 1500
) -> dict:
    """Сокращает отзывы товара только при превышении лимита входа."""
    if count_input_tokens(product, tokenizer) <= max_input_tokens:
        return product.copy()

    prepared_product = product.copy()

    reviews = remove_exact_duplicates(product["reviews"])
    prepared_product["reviews"] = reviews

    if count_input_tokens(prepared_product, tokenizer) <= max_input_tokens:
        return prepared_product

    reviews = remove_similar_reviews(reviews)
    prepared_product["reviews"] = reviews

    if count_input_tokens(prepared_product, tokenizer) <= max_input_tokens:
        return prepared_product

    review_budget = (
        max_input_tokens
        - count_product_tokens_without_reviews(
            prepared_product,
            tokenizer,
        )
    )

    prepared_product["reviews"] = selection_of_reviews(
        review_budget,
        prepared_product,
        tokenizer,
        max_input_tokens,
    )

    return prepared_product

