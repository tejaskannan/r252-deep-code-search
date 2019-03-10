public class Results {

  public boolean checkExists(String path) {
    try {
      if (client.checkExists().forPath(path) != null) {
        return true;
      }
    } catch (Exception e) {
    }
    return false;
  }

  private void doSaveProperties(long version) {
    if (version < lastCacheChanged.get()) {
      return;
    }
    if (file == null) {
      return;
    }
    try {
      File lockfile = new File(file.getAbsolutePath() + ".lock");
      if (!lockfile.exists()) {
        lockfile.createNewFile();
      }
      try (RandomAccessFile raf = new RandomAccessFile(lockfile, "rw");
          FileChannel channel = raf.getChannel()) {
        FileLock lock = channel.tryLock();
        if (lock == null) {
          throw new IOException(
              "can not lock the metadatareport cache file "
                  + file.getAbsolutePath()
                  + ", ignore and retry later, maybe multi java process use the file, please config: dubbo.metadata.file=xxx.properties");
        }
        try {
          if (!file.exists()) {
            file.createNewFile();
          }
          try (FileOutputStream outputFile = new FileOutputStream(file)) {
            properties.store(outputFile, "dubbo metadatareport cache");
          }
        } finally {
          lock.release();
        }
      }
    } catch (Throwable e) {
      if (version < lastCacheChanged.get()) {
        return;
      } else {
        reportCacheExecutor.execute(new SaveProperties(lastCacheChanged.incrementAndGet()));
      }
      logger.warn("failed to save service store file, cause: " + e.getMessage(), e);
    }
  }

  public void doSaveProperties(long version) {
    if (version < lastCacheChanged.get()) {
      return;
    }
    if (file == null) {
      return;
    }
    try {
      File lockfile = new File(file.getAbsolutePath() + ".lock");
      if (!lockfile.exists()) {
        lockfile.createNewFile();
      }
      try (RandomAccessFile raf = new RandomAccessFile(lockfile, "rw");
          FileChannel channel = raf.getChannel()) {
        FileLock lock = channel.tryLock();
        if (lock == null) {
          throw new IOException(
              "can not lock the registry cache file "
                  + file.getAbsolutePath()
                  + ", ignore and retry later, maybe multi java process use the file, please config: dubbo.registry.file=xxx.properties");
        }
        try {
          if (!file.exists()) {
            file.createNewFile();
          }
          try (FileOutputStream outputFile = new FileOutputStream(file)) {
            properties.store(outputFile, "dubbo registry cache");
          }
        } finally {
          lock.release();
        }
      }
    } catch (Throwable e) {
      if (version < lastCacheChanged.get()) {
        return;
      } else {
        registryCacheExecutor.execute(new SaveProperties(lastCacheChanged.incrementAndGet()));
      }
      logger.warn("failed to save registry cache file, cause: " + e.getMessage(), e);
    }
  }

  void loadProperties() {
    if (file != null && file.exists()) {
      try (InputStream in = new FileInputStream(file)) {
        properties.load(in);
        if (logger.isInfoEnabled()) {
          logger.info("load service store file " + file + ", data: " + properties);
        }
      } catch (Throwable e) {
        logger.warn("failed to load service store file " + file, e);
      }
    }
  }

  private void loadProperties() {
    if (file != null && file.exists()) {
      InputStream in = null;
      try {
        in = new FileInputStream(file);
        properties.load(in);
        if (logger.isInfoEnabled()) {
          logger.info("load registry cache file " + file + ", data: " + properties);
        }
      } catch (Throwable e) {
        logger.warn("failed to load registry cache file " + file, e);
      } finally {
        if (in != null) {
          try {
            in.close();
          } catch (IOException e) {
            logger.warn(e.getMessage(), e);
          }
        }
      }
    }
  }

  public boolean checkExists(String path) {
    try {
      return client.exists(path);
    } catch (Throwable t) {
      logger.error("zookeeper failed to check node existing with " + path + ": ", t);
    }
    return false;
  }

  private void check() throws RemotingException {
    long modified = file.lastModified();
    if (modified > last) {
      last = modified;
      changed();
    }
  }

  public FileExchangeGroup(URL url) {
    super(url);
    String path = url.getHost() + "/" + url.getPath();
    file = new File(path);
    if (!file.exists()) {
      throw new IllegalStateException("the group file not exists. file: " + path);
    }
    checkModifiedFuture =
        scheduledExecutorService.scheduleWithFixedDelay(
            new Runnable() {
              @Override
              public void run() {
                try {
                  check();
                } catch (Throwable t) {
                  logger.error("unexpected error occur at reconnect, cause: " + t.getMessage(), t);
                }
              }
            },
            2000,
            2000,
            TimeUnit.MILLISECONDS);
  }

  private void check() throws RemotingException {
    long modified = file.lastModified();
    if (modified > last) {
      last = modified;
      changed();
    }
  }

  protected <T> Invoker<T> doSelect(List<Invoker<T>> invokers, URL url, Invocation invocation) {
    int length = invokers.size();
    boolean sameWeight = true;
    int[] weights = new int[length];
    int firstWeight = getWeight(invokers.get(0), invocation);
    weights[0] = firstWeight;
    int totalWeight = firstWeight;
    for (int i = 1; i < length; i++) {
      int weight = getWeight(invokers.get(i), invocation);
      weights[i] = weight;
      totalWeight += weight;
      if (sameWeight && weight != firstWeight) {
        sameWeight = false;
      }
    }
    if (totalWeight > 0 && !sameWeight) {
      int offset = ThreadLocalRandom.current().nextInt(totalWeight);
      for (int i = 0; i < length; i++) {
        offset -= weights[i];
        if (offset < 0) {
          return invokers.get(i);
        }
      }
    }
    return invokers.get(ThreadLocalRandom.current().nextInt(length));
  }
}
