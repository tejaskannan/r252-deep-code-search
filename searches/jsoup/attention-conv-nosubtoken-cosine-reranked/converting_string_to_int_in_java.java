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
    return String.format("[^%s]", keyPrefix);
  }

  public String toString() {
    return key + "=" + value;
  }

  public W3CBuilder(Document doc) {
    this.doc = doc;
    this.namespacesStack.push(new HashMap<String, String>());
  }

  public T removeHeader(String name) {
    Validate.notEmpty(name, "header name must not be empty");
    Map.Entry<String, List<String>> entry = scanHeaders(name);
    if (entry != null) headers.remove(entry.getKey());
    return (T) this;
  }

  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((value == null) ? 0 : value.hashCode());
    return result;
  }
}
