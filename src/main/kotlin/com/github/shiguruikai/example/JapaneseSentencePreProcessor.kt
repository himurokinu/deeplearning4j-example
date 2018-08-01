package com.github.shiguruikai.example

import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor

class JapaneseSentencePreProcessor : SentencePreProcessor {

    override fun preProcess(sentence: String): String {
        var sb: StringBuilder? = null
        for (index in sentence.indices) {
            var c = sentence[index]
            // 全角英数字を半角に変換
            if (c in '\uff00'..'\uff5e') {
                c -= 0xfee0
            }
            if (c in '\u0021'..'\u002f' || c in '\u003a'..'\u0040' || c in '\u005b'..'\u0060' ||
                    c in '\u007b'..'\u2e79' ||
                    c in '\u3000'..'\u3004' || c in '\u3008'..'\u3040' || c == '・') {
                if (sb == null) {
                    sb = StringBuilder(sentence.length).append(sentence, 0, index).append(' ')
                } else {
                    sb.append(' ')
                }
            } else {
                sb?.append(c)
            }
        }
        return sb?.toString() ?: sentence
    }
}
