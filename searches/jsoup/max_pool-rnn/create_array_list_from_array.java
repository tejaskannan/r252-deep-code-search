public class Results {

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

  public Elements parents() {
    HashSet<Element> combo = new LinkedHashSet<>();
    for (Element e : this) {
      combo.addAll(e.parents());
    }
    return new Elements(combo);
  }

  private static String cacheString(
      final char[] charBuf, final String[] stringCache, final int start, final int count) {
    if (count > maxStringCacheLen) return new String(charBuf, start, count);
    if (count < 1) return "";
    int hash = 0;
    int offset = start;
    for (int i = 0; i < count; i++) {
      hash = 31 * hash + charBuf[offset++];
    }
    final int index = hash & stringCache.length - 1;
    String cached = stringCache[index];
    if (cached == null) {
      cached = new String(charBuf, start, count);
      stringCache[index] = cached;
    } else {
      if (rangeEquals(charBuf, start, count, cached)) {
        return cached;
      } else {
        cached = new String(charBuf, start, count);
        stringCache[index] = cached;
      }
    }
    return cached;
  }

  boolean matchesAnySorted(char[] seq) {
    bufferUp();
    return !isEmpty() && Arrays.binarySearch(seq, charBuf[bufPos]) >= 0;
  }

  public Element prependElement(String tagName) {
    Element child = new Element(Tag.valueOf(tagName, NodeUtils.parser(this).settings()), baseUri());
    prependChild(child);
    return child;
  }

  public List<String> eachAttr(String attributeKey) {
    List<String> attrs = new ArrayList<>(size());
    for (Element element : this) {
      if (element.hasAttr(attributeKey)) attrs.add(element.attr(attributeKey));
    }
    return attrs;
  }

  boolean matchesAny(char... seq) {
    if (isEmpty()) return false;
    bufferUp();
    char c = charBuf[bufPos];
    for (char seek : seq) {
      if (seek == c) return true;
    }
    return false;
  }
}
