import nn.DuelResnetModel
import nn.PlayerMovesIterator
import org.apache.commons.lang3.time.StopWatch
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory
import java.io.File

private const val DATASET_PATH = "/home/bartoszkruba/Desktop/all_with_filtered_anotations_since1998.txt"
private const val SAVE_REFORMATTED_DATASET_TO = "/home/bartoszkruba/Desktop/chess_dataset_reformated.csv"

val log = LoggerFactory.getLogger("Main")

fun main() {
//    prepareData(DATASET_PATH, SAVE_REFORMATTED_DATASET_TO)
    train()
}

private fun train() {
    val stopWatch = StopWatch.create()
    val iterator = PlayerMovesIterator("/home/bartoszkruba/Desktop/chess_moves/chess_moves_au.shuf", 128)
    val model = DuelResnetModel.get(20, 13)

    val scoreIteratorListener = ScoreIterationListener()
    model.setListeners(scoreIteratorListener)

    stopWatch.start()
    var i = 0
    while (iterator.hasNext()) {
        stopWatch.reset()
        stopWatch.start()
        model.fit(iterator.next())
        log.info("Processed batch in ${stopWatch.nanoTime / 1000_000_000.0} seconds")
        if (i % 100 == 0) {
            val test = iterator.next()
            val output = model.output(test.features)
            val predictions = Nd4j.argMax(output[0], 1)
            val labels = Nd4j.argMax(test.labels, 1)
            var right = 0
            for (k in 0 until 128) if (predictions.getInt(k) == labels.getInt(k)) right++
            val accuracy = right / 128.0
            log.info("Accuracy: $accuracy%")
        }
        if (i % 1000 == 0) {
            saveModel(model)
        }
        i++
    }

}

fun saveModel(model: ComputationGraph) {
    log.info("Saving model")
    val currentTime = System.currentTimeMillis()
    val modelPath = File("/home/bartoszkruba/Desktop/models/resnet_chess_$currentTime.zip")
    ModelSerializer.writeModel(model, modelPath, true)
}

