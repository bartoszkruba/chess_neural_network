package nn

import com.github.bhlangonijr.chesslib.*
import org.apache.commons.io.FileUtils
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.factory.Nd4j
import java.io.File

class PlayerMovesIteratorVersion1(private val filePath: String, val batchSize: Int) {

    private var cursor = 1
    private val file = File(filePath)
    private var fileIterator = FileUtils.lineIterator(file)
    private val totalExamples: Int
    val moves = arrayOfNulls<String>(64 * 63).apply {
        var i = 0
        for (s1 in Square.values()) for (s2 in Square.values()) {
            if (s1.toString() != s2.toString() && s1 != Square.NONE && s2 != Square.NONE) {
                this[i] = s1.toString() + s2.toString()
                i++
            }
        }
    }

    init {
        var examples = 0
        file.forEachLine { examples++ }
        totalExamples = examples
        println("total examples: $totalExamples")
    }

    private val figureTypes = setOf(
            Piece.WHITE_PAWN,
            Piece.WHITE_KNIGHT,
            Piece.WHITE_BISHOP,
            Piece.WHITE_ROOK,
            Piece.WHITE_QUEEN,
            Piece.WHITE_KING,
            Piece.BLACK_PAWN,
            Piece.BLACK_KNIGHT,
            Piece.BLACK_BISHOP,
            Piece.BLACK_ROOK,
            Piece.BLACK_QUEEN,
            Piece.BLACK_KING
    )

    private val letterToNumber = hashMapOf("A" to 0, "B" to 1, "C" to 2, "D" to 3, "E" to 4, "F" to 5, "G" to 6, "H" to 7)

    fun next(n: Int): Array<Any> {
        val features = Nd4j.zeros(n, 13, 8, 8)

        // 64 * 63 - all possible move combination between two squares
        // pawn promotion are not taken into consideration
        // no turn counter, no move repetition counter etc..
        val labels = Nd4j.zeros(n, 64 * 63)
        val fens = Array(n) { "" }

        for (example in 0 until n) {
            this.cursor++
            val line = fileIterator.nextLine().split(",")
            val color = line[1]
            val fen = line[2]
            fens[example] = fen
            val move = line[3].substring(0, 4).toUpperCase()
            val board = Board()
            board.loadFromFen(fen)
            for ((counter, figure) in figureTypes.withIndex()) {
                board.getPieceLocation(figure).forEach {
                    val letter = it.toString()[0].toString()
                    val number = it.toString()[1].toString().toInt()

                    // It does not matter for neural network
                    // but the input are formatted so that the board is printed out in the right way
                    features.putScalar(intArrayOf(example, counter, 8 - number, letterToNumber[letter]!!), 1)
                }
            }
            // last plane is reserved for encoding if current player - zeros for white, ones for black
            if (color == "black") for (i in 0 until 8) for (k in 0 until 8) {
                features.putScalar(intArrayOf(example, 12, i, k), 1)
            }
            labels.putScalar(intArrayOf(example, moves.indexOf(move)), 1)
        }

        return arrayOf(features, labels, fens)
    }

    fun next(): Array<Any> = next(batchSize)

    fun reset() {
        fileIterator = FileUtils.lineIterator(file)
    }

    fun hasNext(): Boolean = cursor < (totalExamples - batchSize)
}
