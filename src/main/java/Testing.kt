import com.github.bhlangonijr.chesslib.Board
import com.github.bhlangonijr.chesslib.Square
import com.github.bhlangonijr.chesslib.move.MoveGenerator
import nn.PlayerMovesIteratorVersion1
import org.apache.commons.io.FileUtils
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.factory.Nd4j
import java.io.File

private const val modelPath = "/home/bartoszkruba/Desktop/models/resnet_chess_1588499584329.zip"

fun main() {
    val model = ComputationGraph.load(File(modelPath), true)

    val iterator = PlayerMovesIteratorVersion1("/home/bartoszkruba/Desktop/chess_moves/chess_moves_au.shuf", 128)
    val test = iterator.next()
    val output = model.output(test.features)
    val predictions = Nd4j.argMax(output[0], 1)

    val fileIterator = FileUtils.lineIterator(File("/home/bartoszkruba/Desktop/chess_moves/chess_moves_au.shuf"))

    for (k in 0 until 128) {
        val move = iterator.moves[predictions.getInt(k)]!!
        val from = move.substring(0, 2)
        val to = move.substring(2)
        val board = Board().also { it.loadFromFen(fileIterator.nextLine().split(",")[2]) }
        val legalMoves = MoveGenerator.generateLegalMoves(board)

        if (legalMoves.filter { it.from == Square.fromValue(from) && it.to == Square.fromValue(to) }.firstOrNull() != null)
            println("from: $from | to: $to - legal move")
        else println("from: $from | to: $to - legal move - illegal move")
    }
}