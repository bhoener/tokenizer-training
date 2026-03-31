package tokenizer;
import java.util.*;

public class ReversibleMap<K, V> {
    private Map<K, V> kv;
    private Map<V, K> vk;

    public ReversibleMap() {
        this.kv = new HashMap<>();
        this.vk = new HashMap<>();
    }

    public void putKV(K key, V value) {
        this.kv.put(key, value);
        this.vk.put(value, key);
    }

    public void putVK(V value, K key) {
        this.kv.put(key, value);
        this.vk.put(value, key);
    }

    public V getValue(K key) {
        return this.kv.get(key);
    }

    public K getKey(V value) {
        return this.vk.get(value);
    }

    public boolean containsKey(K key) {
        return this.kv.containsKey(key);
    }

    public boolean containsValue(V value) {
        return this.vk.containsKey(value);
    }

    public Set<K> keySet() {
        return this.kv.keySet();
    }

    public Set<V> valueSet() {
        return this.vk.keySet();
    }

    public Map<K, V> __kv() {
        return this.kv;
    }

    public Map<V, K> __vk() {
        return this.vk;
    }

    @Override
    public String toString() {
        return this.kv.toString();
    }
}
