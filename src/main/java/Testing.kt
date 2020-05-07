import com.github.bhlangonijr.chesslib.Board
import com.github.bhlangonijr.chesslib.Square
import com.github.bhlangonijr.chesslib.move.Move
import com.github.bhlangonijr.chesslib.move.MoveGenerator
import nn.DuelResnetModel
import nn.PlayerMovesIteratorVersion1
import nn.PlayerMovesIteratorVersion2
import org.apache.commons.io.FileUtils
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.factory.Nd4j
import java.io.File

private const val modelPath = "/home/bartoszkruba/Desktop/models/resnet_chess_version_2_1588529014900.zip"

class MoveProbability(val move: Move, val probability: Double)

fun main() {
//    val model = DuelResnetModel.getVersion1(20, 13)
    val model = ComputationGraph.load(File(modelPath), true)

    val iterator = PlayerMovesIteratorVersion2("/home/bartoszkruba/Desktop/chess_moves/chess_moves_au.shuf", 128)
    val test = iterator.next()
    val output = model.output(test[0])
    val fromPredictions = Nd4j.argMax(output[0], 1)
    val toPredictions = Nd4j.argMax(output[1], 1)

    val fileIterator = FileUtils.lineIterator(File("/home/bartoszkruba/Desktop/chess_moves/chess_moves_au.shuf"))

    for (k in 0 until 128) {
        val moveFrom = iterator.squares[fromPredictions.getInt(k)]
        val moveTo = iterator.squares[toPredictions.getInt(k)]
        val split = fileIterator.nextLine().split(",")
        val board = Board().also { it.loadFromFen(split[2]) }
        val legalMoves = MoveGenerator.generateLegalMoves(board)

        println("Move from: $moveFrom | Move to: $moveTo")
    }
}