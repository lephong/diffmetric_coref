package edu.berkeley.nlp.coref.sem
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._
import edu.berkeley.nlp.coref.CorefSystem
import edu.berkeley.nlp.coref.MentionType
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.futile.util.Logger
import scala.collection.mutable.HashSet
import java.io.File
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import java.io.IOException

object QueryCountCollector {
  
  // TODO: Make this more efficient
  // --Some kind of primitive hash map with indexed ints instead of a Counter[(String,String)]?
  // TODO: Think about the casing here...
  def collectCounts(trainPath: String, trainSize: Int, testPath: String, testSize: Int, countsRootDir: String, ngramsPathFile: String) = {
    // Get the head pairs
    val docs = CorefSystem.loadCorefDocs(trainPath, trainSize, null, false) ++ CorefSystem.loadCorefDocs(testPath, testSize, null, false);
    val heads = new HashSet[String]();
    val headPairs = new HashSet[(String, String)];
    for (doc <- docs; i <- 0 until doc.predMentions.size) {
      if (doc.predMentions(i).mentionType != MentionType.PRONOMINAL) {
        heads.add(doc.predMentions(i).headString);
        for (j <- 0 until i) {
          if (doc.predMentions(j).mentionType != MentionType.PRONOMINAL) {
            val first = doc.predMentions(j).headString;
            val second = doc.predMentions(i).headString;
            if (first != second) {
//              Logger.logss("Registering pair: " + first + ", " + second);
              headPairs += first -> second;
            }
          }
        }
      }
    }
    Logger.logss(heads.size + " distinct heads, " + headPairs.size + " distinct head pairs; some of them are " + headPairs.slice(0, Math.min(10, headPairs.size)));
    val headCounts = new Counter[String];
    val headPairCounts = new Counter[(String,String)];
    // Open the outfile early to fail fast
    val outWriter = IOUtils.openOutHard(ngramsPathFile)
    // Load the n-grams and count them
    // Iterate through all 3-grams and 4-grams
    try {
      var numLinesProcessed = 0;
      for (file <- new File(countsRootDir + "/1gms").listFiles) {
        Logger.logss("Processing " + file.getAbsolutePath);
        val lineIterator = IOUtils.lineIterator(file.getAbsolutePath());
        while (lineIterator.hasNext) {
          countUnigram(lineIterator.next, heads, headCounts);
          numLinesProcessed += 1;
        }
      }
      Logger.logss(numLinesProcessed + " 1-grams processed");
      numLinesProcessed = 0;
      for (file <- new File(countsRootDir + "/3gms").listFiles) {
        Logger.logss("Processing " + file.getAbsolutePath);
        val lineIterator = IOUtils.lineIterator(file.getAbsolutePath());
        while (lineIterator.hasNext) {
          count(lineIterator.next, heads, headPairs, headPairCounts, 3);
          numLinesProcessed += 1;
        }
      }
      Logger.logss(numLinesProcessed + " 3-grams processed");
      numLinesProcessed = 0;
      for (file <- new File(countsRootDir + "/4gms").listFiles) {
        Logger.logss("Processing " + file.getAbsolutePath);
        val lineIterator = IOUtils.lineIterator(file.getAbsolutePath());
        while (lineIterator.hasNext) {
          count(lineIterator.next, heads, headPairs, headPairCounts, 4);
          numLinesProcessed += 1;
        }
      }
      Logger.logss(numLinesProcessed + " 4-grams processed");
    } catch {
      case e: IOException => throw new RuntimeException(e);
    }
    // Write to file
    Logger.logss("Extracted counts for " + headCounts.size + " heads and " + headPairCounts.size + " head pairs");
    for (word <- headCounts.keySet.asScala.toSeq.sorted) {
      val str = word + " " + headCounts.getCount(word).toInt
      outWriter.println(str);
    }
    for (pair <- headPairCounts.keySet.asScala.toSeq.sorted) {
      val str = pair._1 + " " + pair._2 + " " + headPairCounts.getCount(pair).toInt
      outWriter.println(str);
//      Logger.logss(str);
    }
    outWriter.close;
  }
  
  def countUnigram(line: String, heads: HashSet[String], headCounts: Counter[String]) {
    val word = fastAccessLine(line, 0, 1);
    if (heads.contains(word)) {
      headCounts.incrementCount(word, fastAccessCount(line));
    }
  }
  
  def count(line: String, heads: HashSet[String], headPairs: HashSet[(String,String)], headPairCounts: Counter[(String, String)], gramSize: Int) {
    val firstWord = fastAccessLine(line, 0, gramSize);
    if (heads.contains(firstWord)) {
      val lastWord = fastAccessLine(line, gramSize - 1, gramSize);
      if (heads.contains(lastWord)) {
        val pair = firstWord -> lastWord;
        val pairFlipped = lastWord -> firstWord;
        if (headPairs.contains(pair) || headPairs.contains(pairFlipped)) {
          if (gramSize == 3) {
            val middleWordLc = fastAccessLine(line, 1, gramSize).toLowerCase;
            if (middleWordLc == "is" || middleWordLc == "are" || middleWordLc == "was" || middleWordLc == "were") {
//              Logger.logss("Matched a pattern: " + line);
              headPairCounts.incrementCount(pair, fastAccessCount(line))
              headPairCounts.incrementCount(pairFlipped, fastAccessCount(line))
            }
          }
          else if (gramSize == 4) {
            val secondWordLc = fastAccessLine(line, 1, gramSize).toLowerCase;
            val thirdWordLc = fastAccessLine(line, 2, gramSize).toLowerCase;
            if ((secondWordLc == "is" || secondWordLc == "are" || secondWordLc == "was" || secondWordLc == "were")
                && (thirdWordLc == "a" || thirdWordLc == "an" || thirdWordLc == "the")) {
//              Logger.logss("Matched a pattern: " + line);
              headPairCounts.incrementCount(pair, fastAccessCount(line))
              headPairCounts.incrementCount(pairFlipped, fastAccessCount(line))
            }
          }
        }
      }
    }
    
  }
  
  def fastAccessLine(line: String, fieldIdx: Int, gramSize: Int) = {
    if (fieldIdx == 0) {
      fastAccessFirst(line);
    } else if (fieldIdx == gramSize - 1) {
      fastAccessLast(line);
    } else {
      fastAccessLineHelper(line, fieldIdx);
    }
  }
  
  def fastAccessCount(line: String) = {
    var firstSpaceIdxFromEnd = line.size - 1;
    while (firstSpaceIdxFromEnd >= 0 && !Character.isWhitespace(line.charAt(firstSpaceIdxFromEnd))) {
      firstSpaceIdxFromEnd -= 1;
    }
    line.slice(firstSpaceIdxFromEnd + 1, line.size).toDouble;
  }
  
  private def fastAccessLineHelper(line: String, fieldIdx: Int) = {
    var wordIdx = 0;
    var inSpace = false;
    var startIdx = 0;
    var endIdx = 0;
    var i = 0;
    while (i < line.size) {
      if (Character.isWhitespace(line.charAt(i))) {
        if (!inSpace) {
          inSpace = true;
          wordIdx += 1;
          if (wordIdx == fieldIdx + 1) {
            endIdx = i;
          }
        }
      } else {
        if (inSpace) {
          inSpace = false;
          if (wordIdx == fieldIdx) {
            startIdx = i;
          }
        }
      }
      i += 1;
    }
    line.slice(startIdx, endIdx);
  }
  
  private def fastAccessFirst(line: String) = {
    var firstSpaceIdx = 0;
    while (firstSpaceIdx < line.size && !Character.isWhitespace(line.charAt(firstSpaceIdx))) {
      firstSpaceIdx += 1;
    }
    line.slice(0, firstSpaceIdx);
  }
  
  private def fastAccessLast(line: String) = {
    // Go past space
    var firstSpaceIdxFromEnd = line.size - 1;
    var count = 0;
    while (firstSpaceIdxFromEnd >= 0 && !Character.isWhitespace(line.charAt(firstSpaceIdxFromEnd))) {
      firstSpaceIdxFromEnd -= 1;
    }
    var endOfWordIdx = firstSpaceIdxFromEnd - 1;
    while (endOfWordIdx >= 0 && Character.isWhitespace(line.charAt(endOfWordIdx))) {
      endOfWordIdx -= 1;
    }
    var beginningOfWordIdx = endOfWordIdx - 1;
    while (beginningOfWordIdx >= 0 && !Character.isWhitespace(line.charAt(beginningOfWordIdx))) {
      beginningOfWordIdx -= 1;
    }
    line.slice(beginningOfWordIdx + 1, endOfWordIdx + 1);
  }
  
  def main(args: Array[String]) {
    val line = "! &quot; </S> 1952";
    println(fastAccessLine(line, 0, 3));
    println(fastAccessLine(line, 1, 3));
    println(fastAccessLine(line, 2, 3));
    println(fastAccessCount(line));
  }
  
}