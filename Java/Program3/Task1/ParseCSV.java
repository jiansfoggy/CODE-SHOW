/**
 * ParseCSV - parse a line from a CSV file, with optional double quotes
 * 
 *  Reference: http://stackoverflow.com/questions/15738918
 */

public class ParseCSV {
  /**
   * parseLine - separate input line into fields at commas, respecting
   *             quoted strings.  We remove quotes around strings.
   * 
   * @param text line from a CSV file
   * @return array of CSV field strings
   */
  static String[] parseLine(String text) {
    String[] split = text.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
    
    for (int i = 0; i < split.length; ++i) {
      String s = split[i];
      if (s.startsWith("\"") && s.endsWith("\"")) {
        split[i] = s.substring(1, s.length() - 1);
      }
    }
    return split;
  }
}
