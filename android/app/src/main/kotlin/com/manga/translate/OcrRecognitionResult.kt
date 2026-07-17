package com.manga.translate

sealed interface OcrRecognitionResult {
    data class Success(val text: String) : OcrRecognitionResult
    data class Failure(val error: Throwable) : OcrRecognitionResult
}

fun OcrRecognitionResult.textOrEmpty(): String {
    return when (this) {
        is OcrRecognitionResult.Success -> text
        is OcrRecognitionResult.Failure -> ""
    }
}

