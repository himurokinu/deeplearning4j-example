package com.github.shiguruikai.example

import com.worksap.nlp.sudachi.DictionaryFactory
import com.worksap.nlp.sudachi.Tokenizer.SplitMode
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator
import org.deeplearning4j.text.stopwords.StopWords
import org.slf4j.LoggerFactory
import java.util.*
import kotlin.system.measureNanoTime

private val topLevelClass = object : Any() {}.javaClass.enclosingClass
private val log = LoggerFactory.getLogger(topLevelClass)

private const val sourcePath = "~/text.txt"
private const val dicDirPath = "~/sudachi/"
private const val dicSettingsPath = "~/sudachi/sudachi_settings.json"
private const val vecOutputPath = "/ramdisk/word2vec.zip"

fun main(args: Array<String>) {

    val sentenceIterator = LineSentenceIterator(sourcePath.toPath().toFile()).apply {
        preProcessor = JapaneseSentencePreProcessor()
    }

    val dictionary = DictionaryFactory().create(
            dicDirPath.toPath().toString(), dicSettingsPath.toPath().toFile().readText())

    fun String.normalize(): String? = when {
        isEmpty() -> null
        first().isUpperCase() -> toLowerCase()
        else -> this
    }

    val tokenizer = SudachiTokenizerFactory(dictionary, SplitMode.B) { morpheme ->
        // 未知語を無視する
        if (morpheme.isOOV) return@SudachiTokenizerFactory null

        // 品詞: 名詞,動詞,形容詞,副詞,形状詞,接尾辞,記号,感動詞,助動詞,補助記号,代名詞,接頭辞,助詞,連体詞,接続詞
        val pos = morpheme.partOfSpeech()
        when (pos[0]) {
            "名詞" -> when {
                pos[1] == "普通名詞" -> when {
                    pos[2] == "副詞可能" -> "_a_"
                    pos[2] == "助数詞可能" -> "_b_"
                    else -> morpheme.normalizedForm()
                }
                pos[1] == "固有名詞" -> morpheme.surface()
                pos[1] == "数詞" -> "_c_"
                else -> morpheme.normalizedForm()
            }
            in "動詞", "形容詞", "代名詞", "形状詞" -> morpheme.normalizedForm()
            else -> return@SudachiTokenizerFactory null
        }.normalize()
    }

    val stopWords: List<String> = (StopWords.getStopWords() +
            ('a'..'f').map { "_" + it + "_" } +
            ('a'..'z').map(Char::toString) +
            ('ぁ'..'ヺ').map(Char::toString) +
            ('ぁ'..'ヺ').map { it + "っ" } +
            ('ぁ'..'ヺ').map { it + "ッ" } +
            "一二三四五六七八九十百千万億兆".map(Char::toString) +
            listOf("する", "もの", "こと", "いる", "よる", "なし", "せい", "つく", "おく", "から", "もん",
                    "為る", "有る", "居る", "成る")).distinct()

    val vec = Word2Vec.Builder()
            .iterations(1)
            .epochs(25)
            .learningRate(0.025)
            .minLearningRate(1e-3)
            .batchSize(1000)
            .layerSize(200)
            .minWordFrequency(5)
            .windowSize(5)
            .stopWords(stopWords)
            .iterate(sentenceIterator)
            .tokenizerFactory(tokenizer)
            .build()

    log.info("学習開始")
    measureNanoTime {
        vec.fit()
    }.let {
        log.info("学習完了: %.3f sec".format(it / 1000_000_000.0))
    }

    dictionary.close()

    log.info("学習データを保存: ${vecOutputPath.toPath().toAbsolutePath()}")
    WordVectorSerializer.writeWord2VecModel(vec, vecOutputPath.toPath().toFile())

    wordsNearest(vec)
    similarity(vec)
    similarWordsInVocabTo(vec)
}

fun wordsNearest(vec: WordVectors) {
    println("関連性のあるワードを表示する")
    val sc = Scanner(System.`in`)
    while (true) {
        print("> ")
        val words = sc.nextLine().split(' ')
        if (words.first() == "exit") {
            break
        } else {
            val positive = mutableListOf<String>()
            val negative = mutableListOf<String>()
            words.forEach {
                when {
                    it.startsWith('+') -> positive += it.substring(1)
                    it.startsWith('-') -> negative += it.substring(1)
                    else -> positive += it
                }
            }
            println(vec.wordsNearest(positive, negative, 10))
            println()
        }
    }
}

fun similarity(vec: WordVectors) {
    println("2つのワードの関連度を表示する")
    val sc = Scanner(System.`in`)
    while (true) {
        print("> ")
        val words = sc.nextLine().split(' ')
        if (words.first() == "exit") {
            break
        } else if (words.size == 2) {
            println(vec.similarity(words[0], words[1]))
        }
        println()
    }
}

fun similarWordsInVocabTo(vec: WordVectors, accuracy: Double = 0.5) {
    println("入力文字を含むワードを検索する")
    val sc = Scanner(System.`in`)
    while (true) {
        print("> ")
        val word = sc.nextLine()
        if (word == "exit") {
            break
        }
        println(vec.similarWordsInVocabTo(word, accuracy))
        println()
    }
}
