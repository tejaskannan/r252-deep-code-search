public class Results {

  protected void runParser() {
    while (true) {
      Token token = tokeniser.read();
      process(token);
      token.reset();
      if (token.type == Token.TokenType.EOF) break;
    }
  }

  static void crossStreams(final InputStream in, final OutputStream out) throws IOException {
    final byte[] buffer = new byte[bufferSize];
    int len;
    while ((len = in.read(buffer)) != -1) {
      out.write(buffer, 0, len);
    }
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

  String consumeToEnd() {
    bufferUp();
    String data = cacheString(charBuf, stringCache, bufPos, bufLength - bufPos);
    bufPos = bufLength;
    return data;
  }

  public static ByteBuffer readToByteBuffer(InputStream inStream, int maxSize) throws IOException {
    Validate.isTrue(maxSize >= 0, "maxsize must be 0 (unlimited) or larger");
    final ConstrainableInputStream input =
        ConstrainableInputStream.wrap(inStream, bufferSize, maxSize);
    return input.readToByteBuffer(maxSize);
  }

  @Override
  void nodelistChanged() {
    super.nodelistChanged();
    shadowChildrenRef = null;
  }

  public void head(Node node, int depth) {
    try {
      node.outerHtmlHead(accum, depth, out);
    } catch (IOException exception) {
      throw new SerializationException(exception);
    }
  }

  String consumeHexSequence() {
    bufferUp();
    int start = bufPos;
    while (bufPos < bufLength) {
      char c = charBuf[bufPos];
      if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'a' && c <= 'f')) bufPos++;
      else break;
    }
    return cacheString(charBuf, stringCache, start, bufPos - start);
  }

  public static Elements collect(Evaluator eval, Element root) {
    Elements elements = new Elements();
    NodeTraversor.traverse(new Accumulator(root, elements, eval), root);
    return elements;
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
}
