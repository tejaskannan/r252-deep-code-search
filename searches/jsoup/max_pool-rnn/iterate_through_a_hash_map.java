public class Results {

  public static void main(String[] args) throws IOException {
    Document doc = Jsoup.connect("http://en.wikipedia.org/").get();
    log(doc.title());
    Elements newsHeadlines = doc.select("#mp-itn b a");
    for (Element headline : newsHeadlines) {
      log("%s\n\t%s", headline.attr("title"), headline.absUrl("href"));
    }
  }

  public static String unescape(String in) {
    StringBuilder out = StringUtil.borrowBuilder();
    char last = 0;
    for (char c : in.toCharArray()) {
      if (c == ESC) {
        if (last != 0 && last == ESC) out.append(c);
      } else out.append(c);
      last = c;
    }
    return StringUtil.releaseBuilder(out);
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

  private static void appendWhitespaceIfBr(Element element, StringBuilder accum) {
    if (element.tag.getName().equals("br") && !TextNode.lastCharIsWhitespace(accum))
      accum.append(" ");
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

  public List<DataNode> dataNodes() {
    List<DataNode> dataNodes = new ArrayList<>();
    for (Node node : childNodes) {
      if (node instanceof DataNode) dataNodes.add((DataNode) node);
    }
    return Collections.unmodifiableList(dataNodes);
  }

  static String mimeBoundary() {
    final StringBuilder mime = StringUtil.borrowBuilder();
    final Random rand = new Random();
    for (int i = 0; i < boundaryLength; i++) {
      mime.append(mimeBoundaryChars[rand.nextInt(mimeBoundaryChars.length)]);
    }
    return StringUtil.releaseBuilder(mime);
  }

  public static Elements select(String query, Iterable<Element> roots) {
    Validate.notEmpty(query);
    Validate.notNull(roots);
    Evaluator evaluator = QueryParser.parse(query);
    ArrayList<Element> elements = new ArrayList<>();
    IdentityHashMap<Element, Boolean> seenElements = new IdentityHashMap<>();
    for (Element root : roots) {
      final Elements found = select(evaluator, root);
      for (Element el : found) {
        if (!seenElements.containsKey(el)) {
          elements.add(el);
          seenElements.put(el, Boolean.TRUE);
        }
      }
    }
    return new Elements(elements);
  }

  @Override
  Tag reset() {
    tagName = null;
    normalName = null;
    pendingAttributeName = null;
    reset(pendingAttributeValue);
    pendingAttributeValueS = null;
    hasEmptyAttributeValue = false;
    hasPendingAttributeValue = false;
    selfClosing = false;
    attributes = null;
    return this;
  }

  private static BomCharset detectCharsetFromBom(final ByteBuffer byteData) {
    final Buffer buffer = byteData;
    buffer.mark();
    byte[] bom = new byte[4];
    if (byteData.remaining() >= bom.length) {
      byteData.get(bom);
      buffer.rewind();
    }
    if (bom[0] == 00 && bom[1] == 00 && bom[2] == (byte) fe && bom[3] == (byte) ff
        || bom[0] == (byte) ff && bom[1] == (byte) fe && bom[2] == 00 && bom[3] == 00) {
      return new BomCharset("utf-32", false);
    } else if (bom[0] == (byte) fe && bom[1] == (byte) ff
        || bom[0] == (byte) ff && bom[1] == (byte) fe) {
      return new BomCharset("utf-16", false);
    } else if (bom[0] == (byte) ef && bom[1] == (byte) bb && bom[2] == (byte) bf) {
      return new BomCharset("utf-8", true);
    }
    return null;
  }
}
