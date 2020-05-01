import com.github.bhlangonijr.chesslib.Board
import com.github.bhlangonijr.chesslib.move.MoveList
import org.apache.commons.io.FileUtils
import java.io.File

fun prepareData(loadFrom: String, saveTo: String) {
    val skipFirst = 5
    var i = 0
    var counter = 0

    val saveReformattedTo = File(saveTo).bufferedWriter()
    FileUtils.lineIterator(File(loadFrom), "UTF-8").use { iterator ->
        while (iterator.hasNext()) {
            i++
            val line = iterator.nextLine()
            if (i > skipFirst) {
                if (counter % 1000 == 0) println(counter)
                val split = line.split(" ")
//                dataset index
                val number = split[0]
//                date of the match
                val date = split[1]
//                result of the match
                val result = split[2]
//                ELO of the white player
                val whiteELO = split[3]
//                ELO of the black player
                val blackELO = split[4]
//                 number of total moves in game
                val len = split[5]
//                is date corrupted or missing?
                val datec = split[6]
//                is result corrupted or missing?
                val resultCorrupted = ("result_true" == split[7])
//                 is white ELO corrupted or missing?
                val weloc = split[8]
//                is black ELO corrupted or missing?
                val beloc = split[9]
//                 is the event date corrupted or missing?
                val edate_c = split[10]
//                is the game initial setup standard or not? (not standard if true)
                val setup = split[11] == "setup_true"
//                 is result properly provided after sequence of moves?
                val fen = split[12]
//                In the original file the result is provided in two places.
//                At the end of each sequence of moves and in the attributes part.
//                This flag indicates if the result is (is not)
                val result2Corrupted = split[13] == "result2_true"
//                is the game date in range [1998, 2007]?
                val oyrange = split[14]
//                is the game length good or not?
                val badLength = split[15] == "blen_true"

                if (!badLength && !result2Corrupted && !result2Corrupted && !setup) {
                    val moves = line.trimEnd().split("### ")[1].split(" ").map { it.split(".")[1] }
                            .joinToString(" ") + " "
                    val moveList = MoveList()
                    moveList.loadFromSan(moves)
                    var turn = 1
                    val board = Board()
                    moveList.forEach {
                        val color = if (turn % 2 != 0) "white" else "black"
                        val fen = board.fen
                        saveReformattedTo.write("$counter,$color,$fen,$it\n")
                        board.doMove(it)
                        turn++
                    }
                }
                counter++
            }
        }
    }
}