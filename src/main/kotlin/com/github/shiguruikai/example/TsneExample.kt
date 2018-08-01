package com.github.shiguruikai.example

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.plot.BarnesHutTsne

private const val vecInputPath = "/ramdisk/word2vec.zip"
private const val csvOutputPath = "/ramdisk/tsne.csv"

fun main(args: Array<String>) {
    val vec = WordVectorSerializer.readWord2VecModel(vecInputPath.toPath().toFile())

    // 任意の単語
    //val labels = listOf("")

    // 100回以上出現する単語のリストを取得
    val labels = vec.vocab.vocabWords().asSequence()
            .filter { it.sequencesCount >= 100 }
            .map { it.word }
            .toList()

    val weights = vec.getWordVectors(labels)

    val tsne = BarnesHutTsne.Builder()
            .setMaxIter(100)
            .perplexity(30.0)
            .theta(0.5)
            .useAdaGrad(false)
            .normalize(false)
            .build()

    tsne.fit(weights)
    tsne.saveAsFile(labels, csvOutputPath.toPath().toString())

    /*
    gnuplot> set datafile separator ","
    gnuplot> plot 'tsne.csv' using 1:2:3 with labels
     */
}
