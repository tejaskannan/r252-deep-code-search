public class Results {

  private static LinkedHashMap<String, List<String>> createHeaderMap(HttpURLConnection conn) {
    final LinkedHashMap<String, List<String>> headers = new LinkedHashMap<>();
    int i = 0;
    while (true) {
      final String key = conn.getHeaderFieldKey(i);
      final String val = conn.getHeaderField(i);
      if (key == null && val == null) break;
      i++;
      if (key == null || val == null) continue;
      if (headers.containsKey(key)) headers.get(key).add(val);
      else {
        final ArrayList<String> vals = new ArrayList<>();
        vals.add(val);
        headers.put(key, vals);
      }
    }
    return headers;
  }

  public List<DataNode> dataNodes() {
    List<DataNode> dataNodes = new ArrayList<>();
    for (Node node : childNodes) {
      if (node instanceof DataNode) dataNodes.add((DataNode) node);
    }
    return Collections.unmodifiableList(dataNodes);
  }

  public List<String> eachAttr(String attributeKey) {
    List<String> attrs = new ArrayList<>(size());
    for (Element element : this) {
      if (element.hasAttr(attributeKey)) attrs.add(element.attr(attributeKey));
    }
    return attrs;
  }

  public Element insertChildren(int index, Collection<? extends Node> children) {
    Validate.notNull(children, "children collection to be inserted must not be null.");
    int currentSize = childNodeSize();
    if (index < 0) index += currentSize + 1;
    Validate.isTrue(index >= 0 && index <= currentSize, "insert position out of bounds.");
    ArrayList<Node> nodes = new ArrayList<>(children);
    Node[] nodeArray = nodes.toArray(new Node[0]);
    addChildren(index, nodeArray);
    return this;
  }

  private ElementMeta createSafeElement(Element sourceEl) {
    String sourceTag = sourceEl.tagName();
    Attributes destAttrs = new Attributes();
    Element dest = new Element(Tag.valueOf(sourceTag), sourceEl.baseUri(), destAttrs);
    int numDiscarded = 0;
    Attributes sourceAttrs = sourceEl.attributes();
    for (Attribute sourceAttr : sourceAttrs) {
      if (whitelist.isSafeAttribute(sourceTag, sourceEl, sourceAttr)) destAttrs.put(sourceAttr);
      else numDiscarded++;
    }
    Attributes enforcedAttrs = whitelist.getEnforcedAttributes(sourceTag);
    destAttrs.addAll(enforcedAttrs);
    return new ElementMeta(dest, numDiscarded);
  }

  protected void initialiseParse(Reader input, String baseUri, Parser parser) {
    super.initialiseParse(input, baseUri, parser);
    state = HtmlTreeBuilderState.Initial;
    originalState = null;
    baseUriSetFromDoc = false;
    headElement = null;
    formElement = null;
    contextElement = null;
    formattingElements = new ArrayList<>();
    pendingTableCharacters = new ArrayList<>();
    emptyEnd = new Token.EndTag();
    framesetOk = true;
    fosterInserts = false;
    fragmentParsing = false;
  }

  public Set<String> classNames() {
    String[] names = classSplit.split(className());
    Set<String> classNames = new LinkedHashSet<>(Arrays.asList(names));
    classNames.remove("");
    return classNames;
  }

  public Element classNames(Set<String> classNames) {
    Validate.notNull(classNames);
    if (classNames.isEmpty()) {
      attributes().remove("class");
    } else {
      attributes().put("class", StringUtil.join(classNames, " "));
    }
    return this;
  }

  boolean matchesIgnoreCase(String seq) {
    bufferUp();
    int scanLength = seq.length();
    if (scanLength > bufLength - bufPos) return false;
    for (int offset = 0; offset < scanLength; offset++) {
      char upScan = Character.toUpperCase(seq.charAt(offset));
      char upTarget = Character.toUpperCase(charBuf[bufPos + offset]);
      if (upScan != upTarget) return false;
    }
    return true;
  }

  public Elements parents() {
    HashSet<Element> combo = new LinkedHashSet<>();
    for (Element e : this) {
      combo.addAll(e.parents());
    }
    return new Elements(combo);
  }
}
