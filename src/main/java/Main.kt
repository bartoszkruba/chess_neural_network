import com.github.bhlangonijr.chesslib.Board
import com.github.bhlangonijr.chesslib.move.MoveList
import org.apache.commons.io.FileUtils
import org.nd4j.linalg.factory.Nd4j
import java.io.File

private const val DATASET_PATH = "/home/bartoszkruba/Desktop/all_with_filtered_anotations_since1998.txt"
private const val SAVE_REFORMATTED_DATASET_TO = "/home/bartoszkruba/Desktop/chess_dataset_reformated.csv"

fun main() {
//    prepareData(DATASET_PATH, SAVE_REFORMATTED_DATASET_TO)
    train()
}


private fun train() {
    val iterator = PlayerMovesIterator("/home/bartoszkruba/Desktop/chess_moves/chess_moves_aa.shuf", 32)
    val dataset = iterator.next()
    val moves = Nd4j.argMax(dataset.labels, 1)
    for (i in 0 until 32) {
        println(iterator.moves[moves.getInt(i)])
    }
}

