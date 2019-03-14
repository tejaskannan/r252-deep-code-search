public class Results {

  public String val() {
    if (size() > 0) return first().val();
    else return "";
  }

  public int hashCode() {
    int result = size;
    result = 31 * result + Arrays.hashCode(keys);
    result = 31 * result + Arrays.hashCode(vals);
    return result;
  }

  public DocumentType(String name, String publicId, String systemId, String baseUri) {
    attr(NAME, name);
    attr(PUBLIC_ID, publicId);
    if (has(PUBLIC_ID)) {
      attr(PUB_SYS_KEY, PUBLIC_KEY);
    }
    attr(SYSTEM_ID, systemId);
  }

  public DocumentType(
      String name, String pubSysKey, String publicId, String systemId, String baseUri) {
    attr(NAME, name);
    if (pubSysKey != null) {
      attr(PUB_SYS_KEY, pubSysKey);
    }
    attr(PUBLIC_ID, publicId);
    attr(SYSTEM_ID, systemId);
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

  public static ConstrainableInputStream wrap(InputStream in, int bufferSize, int maxSize) {
    return in instanceof ConstrainableInputStream
        ? (ConstrainableInputStream) in
        : new ConstrainableInputStream(in, bufferSize, maxSize);
  }

  private ConstrainableInputStream(InputStream in, int bufferSize, int maxSize) {
    super(in, bufferSize);
    Validate.isTrue(maxSize >= 0);
    this.maxSize = maxSize;
    remaining = maxSize;
    capped = maxSize != 0;
    startTime = System.nanoTime();
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

  public static ByteBuffer readToByteBuffer(InputStream inStream, int maxSize) throws IOException {
    Validate.isTrue(maxSize >= 0, "maxsize must be 0 (unlimited) or larger");
    final ConstrainableInputStream input =
        ConstrainableInputStream.wrap(inStream, bufferSize, maxSize);
    return input.readToByteBuffer(maxSize);
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
}
