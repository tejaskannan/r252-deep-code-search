public class Results {

  public static ByteBuffer readToByteBuffer(InputStream inStream, int maxSize) throws IOException {
    Validate.isTrue(maxSize >= 0, "maxsize must be 0 (unlimited) or larger");
    final ConstrainableInputStream input =
        ConstrainableInputStream.wrap(inStream, bufferSize, maxSize);
    return input.readToByteBuffer(maxSize);
  }

  String consumeToEnd() {
    bufferUp();
    String data = cacheString(charBuf, stringCache, bufPos, bufLength - bufPos);
    bufPos = bufLength;
    return data;
  }

  String consumeToAnySorted(final char... chars) {
    bufferUp();
    int pos = bufPos;
    final int start = pos;
    final int remaining = bufLength;
    final char[] val = charBuf;
    while (pos < remaining) {
      if (Arrays.binarySearch(chars, val[pos]) >= 0) break;
      pos++;
    }
    bufPos = pos;
    return bufPos > start ? cacheString(charBuf, stringCache, start, pos - start) : "";
  }

  public Node attr(String key, String value) {
    if (!hasAttributes() && key.equals(nodeName())) {
      this.value = value;
    } else {
      ensureAttributes();
      super.attr(key, value);
    }
    return this;
  }

  public String attr(String key) {
    Validate.notNull(key);
    if (!hasAttributes()) {
      return key.equals(nodeName()) ? (String) value : EmptyString;
    }
    return super.attr(key);
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

  public String consumeToAny(final char... chars) {
    bufferUp();
    int pos = bufPos;
    final int start = pos;
    final int remaining = bufLength;
    final char[] val = charBuf;
    final int charLen = chars.length;
    int i;
    OUTER:
    while (pos < remaining) {
      for (i = 0; i < charLen; i++) {
        if (val[pos] == chars[i]) break OUTER;
      }
      pos++;
    }
    bufPos = pos;
    return pos > start ? cacheString(charBuf, stringCache, start, pos - start) : "";
  }

  public static Elements collect(Evaluator eval, Element root) {
    Elements elements = new Elements();
    NodeTraversor.traverse(new Accumulator(root, elements, eval), root);
    return elements;
  }

  public Connection data(String... keyvals) {
    Validate.notNull(keyvals, "data key value pairs must not be null");
    Validate.isTrue(keyvals.length % 2 == 0, "must supply an even number of key value pairs");
    for (int i = 0; i < keyvals.length; i += 2) {
      String key = keyvals[i];
      String value = keyvals[i + 1];
      Validate.notEmpty(key, "data key must not be empty");
      Validate.notNull(value, "data value must not be null");
      req.data(KeyVal.create(key, value));
    }
    return this;
  }
}
