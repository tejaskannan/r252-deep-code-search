public class Results {

  public String toString() {
    return String.format("[%s^=%s]", key, value);
  }

  public String toString() {
    return String.format("[%s=%s]", key, value);
  }

  public String toString() {
    return String.format("[%s*=%s]", key, value);
  }

  public String toString() {
    return String.format("[%s$=%s]", key, value);
  }

  public String toString() {
    return String.format("[%s!=%s]", key, value);
  }

  public String toString() {
    return String.format(":prev*%s", evaluator);
  }

  public String toString() {
    return String.format(":prev%s", evaluator);
  }

  public String toString() {
    return String.format(":parent%s", evaluator);
  }

  public String toString() {
    return String.format(":not%s", evaluator);
  }

  public String toString() {
    return String.format(":immediateparent%s", evaluator);
  }
}
