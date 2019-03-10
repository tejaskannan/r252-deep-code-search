public class Results {

  static void crossStreams(final InputStream in, final OutputStream out) throws IOException {
    final byte[] buffer = new byte[bufferSize];
    int len;
    while ((len = in.read(buffer)) != -1) {
      out.write(buffer, 0, len);
    }
  }

  public ByteBuffer readToByteBuffer(int max) throws IOException {
    Validate.isTrue(max >= 0, "maxsize must be 0 (unlimited) or larger");
    final boolean localCapped = max > 0;
    final int bufferSize = localCapped && max < DefaultSize ? max : DefaultSize;
    final byte[] readBuffer = new byte[bufferSize];
    final ByteArrayOutputStream outStream = new ByteArrayOutputStream(bufferSize);
    int read;
    int remaining = max;
    while (true) {
      read = read(readBuffer);
      if (read == -1) break;
      if (localCapped) {
        if (read >= remaining) {
          outStream.write(readBuffer, 0, remaining);
          break;
        }
        remaining -= read;
      }
      outStream.write(readBuffer, 0, read);
    }
    return ByteBuffer.wrap(outStream.toByteArray());
  }

  public Connection data(Map<String, String> data) {
    Validate.notNull(data, "data map must not be null");
    for (Map.Entry<String, String> entry : data.entrySet()) {
      req.data(KeyVal.create(entry.getKey(), entry.getValue()));
    }
    return this;
  }

  public Whitelist removeEnforcedAttribute(String tag, String attribute) {
    Validate.notEmpty(tag);
    Validate.notEmpty(attribute);
    TagName tagName = TagName.valueOf(tag);
    if (tagNames.contains(tagName) && enforcedAttributes.containsKey(tagName)) {
      AttributeKey attrKey = AttributeKey.valueOf(attribute);
      Map<AttributeKey, AttributeValue> attrMap = enforcedAttributes.get(tagName);
      attrMap.remove(attrKey);
      if (attrMap.isEmpty()) enforcedAttributes.remove(tagName);
    }
    return this;
  }

  Token read() {
    while (!isEmitPending) state.read(this, reader);
    if (charsBuilder.length() > 0) {
      String str = charsBuilder.toString();
      charsBuilder.delete(0, charsBuilder.length());
      charsString = null;
      return charPending.data(str);
    } else if (charsString != null) {
      Token token = charPending.data(charsString);
      charsString = null;
      return token;
    } else {
      isEmitPending = false;
      return emitPending;
    }
  }

  private void prepareByteData() {
    Validate.isTrue(
        executed,
        "request must be executed (with .execute(), .get(), or .post() before getting response body");
    if (byteData == null) {
      Validate.isFalse(inputStreamRead, "request has already been read (with .parse())");
      try {
        byteData = DataUtil.readToByteBuffer(bodyStream, req.maxBodySize());
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      } finally {
        inputStreamRead = true;
        safeClose();
      }
    }
  }

  public static ByteBuffer readToByteBuffer(InputStream inStream, int maxSize) throws IOException {
    Validate.isTrue(maxSize >= 0, "maxsize must be 0 (unlimited) or larger");
    final ConstrainableInputStream input =
        ConstrainableInputStream.wrap(inStream, bufferSize, maxSize);
    return input.readToByteBuffer(maxSize);
  }

  void rewindToMark() {
    if (bufMark == -1) throw new UncheckedIOException(new IOException("mark invalid"));
    bufPos = bufMark;
  }

  public String attr(String attributeKey) {
    Validate.notNull(attributeKey);
    if (!hasAttributes()) return EmptyString;
    String val = attributes().getIgnoreCase(attributeKey);
    if (val.length() > 0) return val;
    else if (attributeKey.startsWith("abs:"))
      return absUrl(attributeKey.substring("abs:".length()));
    else return "";
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
}
