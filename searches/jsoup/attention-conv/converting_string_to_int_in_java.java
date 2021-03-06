public class Results {

  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((value == null) ? 0 : value.hashCode());
    return result;
  }

  public int hashCode() {
    int result = key != null ? key.hashCode() : 0;
    result = 31 * result + (val != null ? val.hashCode() : 0);
    return result;
  }

  public Element val(String value) {
    if (tagName().equals("textarea")) text(value);
    else attr("value", value);
    return this;
  }

  public T removeHeader(String name) {
    Validate.notEmpty(name, "header name must not be empty");
    Map.Entry<String, List<String>> entry = scanHeaders(name);
    if (entry != null) headers.remove(entry.getKey());
    return (T) this;
  }

  public Node attr(String attributeKey, String attributeValue) {
    attributeKey = NodeUtils.parser(this).settings().normalizeAttribute(attributeKey);
    attributes().putIgnoreCase(attributeKey, attributeValue);
    return this;
  }

  public String val() {
    if (tagName().equals("textarea")) return text();
    else return attr("value");
  }

  public W3CBuilder(Document doc) {
    this.doc = doc;
    this.namespacesStack.push(new HashMap<String, String>());
  }

  public Element attr(String attributeKey, String attributeValue) {
    super.attr(attributeKey, attributeValue);
    return this;
  }

  public String toString() {
    return String.format("[%s*=%s]", key, value);
  }

  public String toString() {
    return String.format("[%s^=%s]", key, value);
  }
}
