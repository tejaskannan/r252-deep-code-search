public class Results {

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

  public String toString() {
    return String.format(":prev%s", evaluator);
  }

  public String toString() {
    return String.format(":immediateparent%s", evaluator);
  }

  public String toString() {
    if (a == 0) return String.format(":%s(%d)", getPseudoClass(), b);
    if (b == 0) return String.format(":%s(%dn)", getPseudoClass(), a);
    return String.format(":%s(%dn%+d)", getPseudoClass(), a, b);
  }

  public String toString() {
    return String.format("[%s*=%s]", key, value);
  }

  public String toString() {
    return String.format("[%s^=%s]", key, value);
  }

  public String toString() {
    return String.format("[%s=%s]", key, value);
  }
}
