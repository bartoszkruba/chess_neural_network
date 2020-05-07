import com.github.bhlangonijr.chesslib.Board
import com.github.bhlangonijr.chesslib.move.Move
import com.github.bhlangonijr.chesslib.move.MoveGenerator
import nn.DuelResnetModel
import nn.PlayerMovesIteratorVersion1
import nn.PlayerMovesIteratorVersion2
import org.apache.commons.lang3.time.StopWatch
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Sgd
import org.slf4j.LoggerFactory
import java.io.File

private const val DATASET_PATH = "/home/bartoszkruba/Desktop/all_with_filtered_anotations_since1998.txt"
private const val SAVE_REFORMATTED_DATASET_TO = "/home/bartoszkruba/Desktop/chess_dataset_reformated.csv"

//class MoveProbability(val move: Move, val probability: Double)

val log = LoggerFactory.getLogger("Main")

fun main() {
//    prepareData(DATASET_PATH, SAVE_REFORMATTED_DATASET_TO)
    trainVersionOne()
//    trainVersionTwo()
}

private fun trainVersionOne() {
    val stopWatch = StopWatch.create()
    val iterator = PlayerMovesIteratorVersion1("/home/bartoszkruba/Desktop/chess_moves/chess_moves_ba.shuf", 128)
    val model = ComputationGraph.load(File("/home/bartoszkruba/Desktop/models/resnet_chess_1588732752320.zip"), true)
//    val model = DuelResnetModel.getVersion1(20, 13)

    val scoreIteratorListener = ScoreIterationListener()
    model.setListeners(scoreIteratorListener)

    stopWatch.start()
    var i = 1
    while (iterator.hasNext()) {
        try {
            stopWatch.reset()
            stopWatch.start()
            val dataset = iterator.next()
            model.fit(arrayOf(dataset[0] as INDArray), arrayOf(dataset[1] as INDArray))
            log.info("$i - Processed batch in ${stopWatch.nanoTime / 1000_000_000.0} seconds")
            if (i % 100 == 0) {
                val test = iterator.next()
                val output = model.output(test[0] as INDArray)
                val labels = Nd4j.argMax(test[1] as INDArray, 1)
                var right = 0
                for (k in 0 until 128) {
                    val probabilities = ArrayList<MoveProbability>()
                    val board = Board().apply { loadFromFen((test[2] as Array<String>)[k]) }
                    val legalMoves = MoveGenerator.generateLegalMoves(board)
                    legalMoves.forEach {
                        val legalMove = (it.from.toString() + it.to.toString()).toUpperCase()
                        val probability = output[0].getDouble(0, iterator.moves.indexOf(legalMove))
                        probabilities.add(MoveProbability(it, probability * 100))
                    }
                    probabilities.sortWith(compareByDescending { it.probability })
                    val rightMove = iterator.moves[labels.getInt(k)]
                    if (probabilities.first().move.toString().substring(0, 4).toUpperCase()
                            == rightMove.toString().substring(0, 4).toUpperCase())
                        right++
                }
                val accuracy = right / 128.0
                log.info("Accuracy: ${accuracy * 10}%")
            }
            if (i % 1000 == 0) {
                saveModel(model, "resnet_chess")
            }
            i++
        } catch (ex: Exception) {
            ex.printStackTrace()
        }
    }
}

private fun trainVersionTwo() {
    val stopWatch = StopWatch.create()
    val iterator = PlayerMovesIteratorVersion2("/home/bartoszkruba/Desktop/chess_moves/chess_moves_au.shuf", 128)
    val model = DuelResnetModel.getVersion2(20, 13)

    val scoreIteratorListener = ScoreIterationListener(1)
    model.setListeners(scoreIteratorListener)

    var i = 1
    while (iterator.hasNext()) {
        stopWatch.reset()
        stopWatch.start()
        val dataset = iterator.next()
        model.fit(arrayOf(dataset[0]), arrayOf(dataset[1], dataset[2]))
        log.info("$i - Processed batch in ${stopWatch.nanoTime / 1000_000_000.0} seconds")
        if (i % 100 == 0) {
            val test = iterator.next()
            val output = model.output(test[0])
            val fromPredictions = Nd4j.argMax(output[0], 1)
            val toPredictions = Nd4j.argMax(output[1], 1)
            val fromLabels = Nd4j.argMax(test[1], 1)
            val toLabels = Nd4j.argMax(test[2], 1)

            var right = 0
            for (k in 0 until 128) if (fromPredictions.getInt(k) == fromLabels.getInt(k)) right++
            var accuracy = right / 128.0
            log.info("From Accuracy: ${accuracy * 10}%")

            right = 0
            for (k in 0 until 128) if (toPredictions.getInt(k) == toLabels.getInt(k)) right++
            accuracy = right / 128.0
            log.info("To Accuracy: ${accuracy * 10}%")
        }
        if (i % 1000 == 0) {
            saveModel(model, "resnet_chess_version_2")
        }
        i++
    }
}

fun saveModel(model: ComputationGraph, name: String) {
    log.info("Saving model")
    val currentTime = System.currentTimeMillis()
    val modelPath = File("/home/bartoszkruba/Desktop/models/${name}_$currentTime.zip")
    ModelSerializer.writeModel(model, modelPath, true)
}

