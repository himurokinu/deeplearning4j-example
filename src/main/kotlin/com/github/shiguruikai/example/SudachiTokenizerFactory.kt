package com.github.shiguruikai.example

import com.worksap.nlp.sudachi.Dictionary
import com.worksap.nlp.sudachi.Morpheme
import com.worksap.nlp.sudachi.Tokenizer.SplitMode
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
import java.io.InputStream

class SudachiTokenizerFactory(
        private val dictionary: Dictionary,
        private val splitMode: SplitMode,
        private val transform: ((Morpheme) -> String?)? = null
) : TokenizerFactory {

    private var preProcess: TokenPreProcess? = null

    override fun create(toTokenize: String): Tokenizer {
        return SudachiTokenizer(toTokenize, dictionary, splitMode, transform).also {
            it.setTokenPreProcessor(preProcess)
        }
    }

    override fun create(toTokenize: InputStream): Tokenizer {
        throw UnsupportedOperationException()
    }

    override fun setTokenPreProcessor(tokenPreProcess: TokenPreProcess?) {
        this.preProcess = tokenPreProcess
    }

    override fun getTokenPreProcessor(): TokenPreProcess? = preProcess
}
