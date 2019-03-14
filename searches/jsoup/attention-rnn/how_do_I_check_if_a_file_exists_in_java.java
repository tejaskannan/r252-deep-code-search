public class Results {

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

  public Elements parents() {
    Elements parents = new Elements();
    accumulateParents(this, parents);
    return parents;
  }

  public static URL resolve(URL base, String relUrl) throws MalformedURLException {
    if (relUrl.startsWith("?")) relUrl = base.getPath() + relUrl;
    if (relUrl.indexOf('.') == 0 && base.getFile().indexOf('/') != 0) {
      base = new URL(base.getProtocol(), base.getHost(), base.getPort(), "/" + base.getFile());
    }
    return new URL(base, relUrl);
  }

  public Connection data(String key, String value) {
    req.data(KeyVal.create(key, value));
    return this;
  }

  static void crossStreams(final InputStream in, final OutputStream out) throws IOException {
    final byte[] buffer = new byte[bufferSize];
    int len;
    while ((len = in.read(buffer)) != -1) {
      out.write(buffer, 0, len);
    }
  }

  public Connection data(Map<String, String> data) {
    Validate.notNull(data, "data map must not be null");
    for (Map.Entry<String, String> entry : data.entrySet()) {
      req.data(KeyVal.create(entry.getKey(), entry.getValue()));
    }
    return this;
  }

  public Whitelist removeTags(String... tags) {
    Validate.notNull(tags);
    for (String tag : tags) {
      Validate.notEmpty(tag);
      TagName tagName = TagName.valueOf(tag);
      if (tagNames.remove(tagName)) {
        attributes.remove(tagName);
        enforcedAttributes.remove(tagName);
        protocols.remove(tagName);
      }
    }
    return this;
  }

  public void head(Node node, int depth) {
    String name = node.nodeName();
    if (node instanceof TextNode) append(((TextNode) node).text());
    else if (name.equals("li")) append("\n * ");
    else if (name.equals("dt")) append("  ");
    else if (StringUtil.in(name, "p", "h1", "h2", "h3", "h4", "h5", "tr")) append("\n");
  }

  private static boolean needsMultipart(Connection.Request req) {
    for (Connection.KeyVal keyVal : req.data()) {
      if (keyVal.hasInputStream()) return true;
    }
    return false;
  }

  public void head(Node node, int depth) {
    if (node instanceof TextNode) {
      TextNode textNode = (TextNode) node;
      accum.append(textNode.getWholeText());
    }
  }
}
