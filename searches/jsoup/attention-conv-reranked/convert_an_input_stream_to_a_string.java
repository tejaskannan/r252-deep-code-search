public class Results {

  public static ConstrainableInputStream wrap(InputStream in, int bufferSize, int maxSize) {
    return in instanceof ConstrainableInputStream
        ? (ConstrainableInputStream) in
        : new ConstrainableInputStream(in, bufferSize, maxSize);
  }

  public String attr(String key) {
    Validate.notNull(key);
    if (!hasAttributes()) {
      return key.equals(nodeName()) ? (String) value : EmptyString;
    }
    return super.attr(key);
  }

  public String body() {
    prepareByteData();
    String body;
    if (charset == null)
      body = Charset.forName(DataUtil.defaultCharset).decode(byteData).toString();
    else body = Charset.forName(charset).decode(byteData).toString();
    ((Buffer) byteData).rewind();
    return body;
  }

  public Connection submit() {
    String action = hasAttr("action") ? absUrl("action") : baseUri();
    Validate.notEmpty(
        action,
        "could not determine a form action url for submit. ensure you set a base uri when parsing.");
    Connection.Method method =
        attr("method").toUpperCase().equals("post")
            ? Connection.Method.POST
            : Connection.Method.GET;
    return Jsoup.connect(action).data(formData()).method(method);
  }

  public int hashCode() {
    int result = size;
    result = 31 * result + Arrays.hashCode(keys);
    result = 31 * result + Arrays.hashCode(vals);
    return result;
  }

  public int read(byte[] b, int off, int len) throws IOException {
    if (interrupted || capped && remaining <= 0) return -1;
    if (Thread.interrupted()) {
      interrupted = true;
      return -1;
    }
    if (expired()) throw new SocketTimeoutException("read timeout");
    if (capped && len > remaining) len = remaining;
    try {
      final int read = super.read(b, off, len);
      remaining -= read;
      return read;
    } catch (SocketTimeoutException e) {
      return 0;
    }
  }

  public int hashCode() {
    int result = tagName.hashCode();
    result = 31 * result + (isBlock ? 1 : 0);
    result = 31 * result + (formatAsBlock ? 1 : 0);
    result = 31 * result + (canContainInline ? 1 : 0);
    result = 31 * result + (empty ? 1 : 0);
    result = 31 * result + (selfClosing ? 1 : 0);
    result = 31 * result + (preserveWhitespace ? 1 : 0);
    result = 31 * result + (formList ? 1 : 0);
    result = 31 * result + (formSubmit ? 1 : 0);
    return result;
  }

  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((value == null) ? 0 : value.hashCode());
    return result;
  }

  public Node attr(String attributeKey, String attributeValue) {
    attributeKey = NodeUtils.parser(this).settings().normalizeAttribute(attributeKey);
    attributes().putIgnoreCase(attributeKey, attributeValue);
    return this;
  }

  public Elements attr(String attributeKey, String attributeValue) {
    for (Element element : this) {
      element.attr(attributeKey, attributeValue);
    }
    return this;
  }
}
