public class Results {

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

  public static ByteBuffer readToByteBuffer(InputStream inStream, int maxSize) throws IOException {
    Validate.isTrue(maxSize >= 0, "maxsize must be 0 (unlimited) or larger");
    final ConstrainableInputStream input =
        ConstrainableInputStream.wrap(inStream, bufferSize, maxSize);
    return input.readToByteBuffer(maxSize);
  }

  static void crossStreams(final InputStream in, final OutputStream out) throws IOException {
    final byte[] buffer = new byte[bufferSize];
    int len;
    while ((len = in.read(buffer)) != -1) {
      out.write(buffer, 0, len);
    }
  }

  protected void runParser() {
    while (true) {
      Token token = tokeniser.read();
      process(token);
      token.reset();
      if (token.type == Token.TokenType.EOF) break;
    }
  }

  public static void main(String... args) throws IOException {
    Validate.isTrue(
        args.length == 1 || args.length == 2,
        "usage: java -cp jsoup.jar org.jsoup.examples.htmltoplaintext url [selector]");
    final String url = args[0];
    final String selector = args.length == 2 ? args[1] : null;
    Document doc = Jsoup.connect(url).userAgent(userAgent).timeout(timeout).get();
    HtmlToPlainText formatter = new HtmlToPlainText();
    if (selector != null) {
      Elements elements = doc.select(selector);
      for (Element element : elements) {
        String plainText = formatter.getPlainText(element);
        System.out.println(plainText);
      }
    } else {
      String plainText = formatter.getPlainText(doc);
      System.out.println(plainText);
    }
  }

  public static void main(String[] args) throws IOException {
    Validate.isTrue(args.length == 1, "usage: supply url to fetch");
    String url = args[0];
    print("fetching %s...", url);
    Document doc = Jsoup.connect(url).get();
    Elements links = doc.select("a[href]");
    Elements media = doc.select("[src]");
    Elements imports = doc.select("link[href]");
    print("\nmedia: (%d)", media.size());
    for (Element src : media) {
      if (src.tagName().equals("img"))
        print(
            " * %s: <%s> %sx%s (%s)",
            src.tagName(),
            src.attr("abs:src"),
            src.attr("width"),
            src.attr("height"),
            trim(src.attr("alt"), 20));
      else print(" * %s: <%s>", src.tagName(), src.attr("abs:src"));
    }
    print("\nimports: (%d)", imports.size());
    for (Element link : imports) {
      print(" * %s <%s> (%s)", link.tagName(), link.attr("abs:href"), link.attr("rel"));
    }
    print("\nlinks: (%d)", links.size());
    for (Element link : links) {
      print(" * a: <%s>  (%s)", link.attr("abs:href"), trim(link.text(), 35));
    }
  }

  String consumeToEnd() {
    bufferUp();
    String data = cacheString(charBuf, stringCache, bufPos, bufLength - bufPos);
    bufPos = bufLength;
    return data;
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
}
