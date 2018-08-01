package com.github.shiguruikai.example

import com.worksap.nlp.sudachi.Dictionary
import com.worksap.nlp.sudachi.Morpheme
import com.worksap.nlp.sudachi.Tokenizer.SplitMode
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer

class SudachiTokenizer(
        text: String,
        dictionary: Dictionary,
        splitMode: SplitMode,
        transform: ((Morpheme) -> String?)? = null
) : Tokenizer {

    companion object {
        private val emptyStringList = listOf<String>()
    }

    private val tokens: List<String>
    private var nextIndex: Int = 0
    private var preProcess: TokenPreProcess? = null

    init {
        tokens = if (text.isEmpty()) {
            emptyStringList
        } else {
            val tokenizer = dictionary.create()
            val sequence = tokenizer.tokenize(splitMode, text)
            if (transform != null) {
                sequence.mapNotNull { transform(it) }
            } else {
                sequence.mapNotNull { it.surface() }
            }
        }
    }

    private fun getNextToken(): String {
        val token = tokens[nextIndex++]
        return preProcess?.preProcess(token) ?: token
    }

    override fun countTokens(): Int = tokens.size

    override fun hasMoreTokens(): Boolean = nextIndex in 0 until tokens.size

    override fun nextToken(): String {
        if (!hasMoreTokens()) throw NoSuchElementException()

        return getNextToken()
    }

    override fun getTokens(): List<String> = MutableList(tokens.size - nextIndex) { getNextToken() }

    override fun setTokenPreProcessor(tokenPreProcess: TokenPreProcess?) {
        this.preProcess = tokenPreProcess
    }
}
