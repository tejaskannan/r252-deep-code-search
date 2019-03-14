public class Results {

  public String toString() {
    return String.format("[%s*=%s]", key, value);
  }

  public String toString() {
    return String.format("[%s^=%s]", key, value);
  }

  public String toString() {
    return String.format("[%s=%s]", key, value);
  }

  public String toString() {
    return String.format("[%s!=%s]", key, value);
  }

  public String toString() {
    return String.format("[%s$=%s]", key, value);
  }

  void outerHtmlHead(Appendable accum, int depth, Document.OutputSettings out) throws IOException {
    accum.append("<").append(isProcessingInstruction ? "!" : "?").append(coreValue());
    getWholeDeclaration(accum, out);
    accum.append(isProcessingInstruction ? "!" : "?").append(">");
  }

  public String toString() {
    return String.format(":has(%s)", evaluator);
  }

  public String toString() {
    return String.format(":prev*%s", evaluator);
  }

  public String toString() {
    return String.format(":parent%s", evaluator);
  }

  public String toString() {
    return String.format(":not%s", evaluator);
  }
}
