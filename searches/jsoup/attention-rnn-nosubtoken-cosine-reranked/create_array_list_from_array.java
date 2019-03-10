public class Results {

  public List<String> eachAttr(String attributeKey) {
    List<String> attrs = new ArrayList<>(size());
    for (Element element : this) {
      if (element.hasAttr(attributeKey)) attrs.add(element.attr(attributeKey));
    }
    return attrs;
  }

  public List<Node> childNodesCopy() {
    final List<Node> nodes = ensureChildNodes();
    final ArrayList<Node> children = new ArrayList<>(nodes.size());
    for (Node node : nodes) {
      children.add(node.clone());
    }
    return children;
  }

  public List<Node> siblingNodes() {
    if (parentNode == null) return Collections.emptyList();
    List<Node> nodes = parentNode.ensureChildNodes();
    List<Node> siblings = new ArrayList<>(nodes.size() - 1);
    for (Node node : nodes) if (node != this) siblings.add(node);
    return siblings;
  }

  public Node unwrap() {
    Validate.notNull(parentNode);
    final List<Node> childNodes = ensureChildNodes();
    Node firstChild = childNodes.size() > 0 ? childNodes.get(0) : null;
    parentNode.addChildren(siblingIndex, this.childNodesAsArray());
    this.remove();
    return firstChild;
  }

  public Node attr(String attributeKey, String attributeValue) {
    attributeKey = NodeUtils.parser(this).settings().normalizeAttribute(attributeKey);
    attributes().putIgnoreCase(attributeKey, attributeValue);
    return this;
  }

  public <T extends Appendable> T html(T appendable) {
    final int size = childNodes.size();
    for (int i = 0; i < size; i++) childNodes.get(i).outerHtml(appendable);
    return appendable;
  }

  public Elements clone() {
    Elements clone = new Elements(size());
    for (Element e : this) clone.add(e.clone());
    return clone;
  }

  public Element attr(String attributeKey, boolean attributeValue) {
    attributes().put(attributeKey, attributeValue);
    return this;
  }

  String consumeToEnd() {
    bufferUp();
    String data = cacheString(charBuf, stringCache, bufPos, bufLength - bufPos);
    bufPos = bufLength;
    return data;
  }

  public Elements attr(String attributeKey, String attributeValue) {
    for (Element element : this) {
      element.attr(attributeKey, attributeValue);
    }
    return this;
  }
}
