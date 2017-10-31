/** Count number of occurences in a text file
 * 
 * @author thaianthantrong
 *
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class FirstCount {
	
	private static final Set<String> VALUES = new HashSet<String>(Arrays.asList(
		     new String[] {"le","la","les","on"}
		));

	public static void main(String[] args) throws IOException {

		System.out.println("--- WORDCOUNT ---");
		long startTime = System.currentTimeMillis();

		String fileName = "CC-MAIN-20170322212949-00140-ip-10-233-31-227.ec2.internal.warc.wet";

		FileReader in = new FileReader(fileName);
		BufferedReader br = new BufferedReader(in);
	
		Map<String, Integer> hm = new HashMap<String, Integer>();

		String line;
		while ((line = br.readLine()) != null) {
			// System.out.println(line);
			
			// Remove special characters
			String line_adjusted = line.replaceAll("[^\\x00-\\x7f]+", "");
			
			String[] words = line_adjusted.split(" ");

			for (int i = 0; i < words.length; i++) {
				String word_lowerCased = words[i].toLowerCase();
				boolean contains = VALUES.contains(word_lowerCased);
				
				if (!contains){
					if (!hm.containsKey(word_lowerCased)) {
						hm.put(word_lowerCased, 1);
					} else {
						int cont = hm.get(word_lowerCased);
						hm.put(word_lowerCased, cont + 1);
					}
				}
			}
		}
		in.close();

		System.out.println(" ");
		System.out.println("--- SORTED ---");
		List<Map.Entry<String, Integer>> list = new ArrayList<Map.Entry<String, Integer>>(hm.entrySet());
		Collections.sort(list, new ValueThenKeyComparator<String, Integer>());
		
		// System.out.println(list);
		for (int j=0; j<50; j++){ 
			System.out.println(list.get(j).getKey() + " " + list.get(j).getValue());
		}
		
		long endTime   = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		
		System.out.println("--- TIME ---");
		System.out.println(totalTime + " ms");
	
	}

}

class ValueThenKeyComparator<K extends Comparable<? super K>, V extends Comparable<? super V>>
		implements Comparator<Map.Entry<K, V>> {

	public int compare(Map.Entry<K, V> a, Map.Entry<K, V> b) {
		int cmp1 = b.getValue().compareTo(a.getValue());
		if (cmp1 != 0) {
			return cmp1;
		} else {
			return a.getKey().compareTo(b.getKey());
		}
	}
}
