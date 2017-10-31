package PartNine;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class MASTER {
	private final static String COMPUTER = "/Users/thaianthantrong/Documents/MS_BIG_DATA/Cours/INF727/tathan";

	private String fileDispos;

	private String[] computers;

	private final static int COUNT = 3;

	public MASTER(String fileDispos) throws IOException {
		this.fileDispos = fileDispos;
		this.computers = new String[COUNT];
		FileReader in = new FileReader(this.fileDispos);
		BufferedReader br = new BufferedReader(in);

		String line;
		int count = 0;
		while ((count < COUNT) && (line = br.readLine()) != null) {
			this.computers[count] = line;
			count++;
		}
		br.close();
	}

	public Map<String, String> getMapsComputer() {
		Map<String, String> hm = new HashMap<String, String>();

		for (int i = 0; i < COUNT; i++) {
			hm.put("UM" + i, this.computers[i]);
		}
		return hm;
	}

	public void startThreads(ProcessBuilder pb) throws IOException {
		Process p = pb.start();

		BufferedReader reader = new BufferedReader(new InputStreamReader(
				p.getInputStream()));

		BufferedReader reader_err = new BufferedReader(new InputStreamReader(
				p.getErrorStream()));

		MyThread input = new MyThread(reader);
		input.start();

		MyThread input_err = new MyThread(reader_err);
		input_err.start();
	}

	public String getFileDispos() {
		return this.fileDispos;
	}

	public void pasteSplits() throws IOException {

		int count = 0;

		while (count < COUNT) { // FOR EACH COMPUTER
			ProcessBuilder[] pb_scps = new ProcessBuilder[COUNT];
			// System.out.println("COMPUTER : " + this.computers[count]);

			for (int i = 0; i < COUNT; i++) { // COPY AND PASTE EACH SX.TXT FILE
				ProcessBuilder pb_scp = new ProcessBuilder("scp", COMPUTER
						+ "/splits/s" + i + ".txt", "tathan@"
						+ this.computers[count] + ":/tmp/tathan/splits/s" + i
						+ ".txt");

				pb_scps[i] = pb_scp;
			}
			count++;

			this.startThreads(pb_scps[0]);
			this.startThreads(pb_scps[1]);
			this.startThreads(pb_scps[2]);
		}
		// br.close();
	}

	public void startKeys(ProcessBuilder pb, Map<String, ArrayList<String>> hm,
			String um) throws IOException {
		Process p = pb.start();

		BufferedReader reader = new BufferedReader(new InputStreamReader(
				p.getInputStream()));

		Keys input = new Keys(reader, hm, um);
		input.start();

	}

	public void buildUMs() throws IOException {

		int count = 0;

		/* QUESTION 52 : CREATE HASHMAP TO STORE KEYS : <KEY, <UMx, UMy>> */
		Map<String, ArrayList<String>> hm = new HashMap<String, ArrayList<String>>();

		while ((count < COUNT)) {
			ProcessBuilder[] pb_slaves = new ProcessBuilder[COUNT];

			/* GO TO COMPUTER */
			System.out.println("COMPUTER : " + this.computers[count]);

			/* RUN SLAVE. JAR */

			for (int i = 0; i < COUNT; i++) {

				System.out.println("	| Run slave jar on file s" + i
						+ ".txt with " + this.computers[count]);
				ProcessBuilder pb_slave = new ProcessBuilder("ssh", "-o",
						"StrictHostKeyChecking=no", "tathan@"
								+ this.computers[count], "java", "-jar",
						"/tmp/tathan/slave.jar", "0", "/tmp/tathan/splits/s"
								+ i + ".txt");

				pb_slaves[i] = pb_slave;
				// this.startKeys(pb_slave, hm, "UM" + count);

			}
			this.startKeys(pb_slaves[0], hm, "UM" + 0);
			this.startKeys(pb_slaves[1], hm, "UM" + 1);
			this.startKeys(pb_slaves[2], hm, "UM" + 2);

			count++;

		}

		// br.close();
	}

	public String toString(Map<String, ArrayList<String>> hm) {
		String s = "";
		Iterator<String> itr = hm.keySet().iterator();

		while (itr.hasNext()) {
			String key = itr.next();
			s += key + "< ";
			Iterator<String> itr_val = hm.get(key).iterator();
			while (itr_val.hasNext()) {
				String val = itr_val.next();
				if (itr_val.hasNext()) {
					s += val + ", ";
				} else {
					s += val;
				}
			}
			s += " > \n";
		}

		return s;
	}

	public Map<String, ArrayList<String>> waitForMap()
			throws InterruptedException {
		BlockingQueue<Map<String, ArrayList<String>>> queue = new ArrayBlockingQueue<Map<String, ArrayList<String>>>(
				10);
		Map<String, ArrayList<String>> hm = new HashMap<String, ArrayList<String>>();

		Producer producer1 = new Producer(queue, hm, this.computers[0], 0);
		Producer producer2 = new Producer(queue, hm, this.computers[1], 1);
		Producer producer3 = new Producer(queue, hm, this.computers[2], 2);

		Consumer consumer1 = new Consumer(queue);
		// Consumer consumer2 = new Consumer(queue);

		Thread t1 = new Thread(producer1);
		t1.start();

		Thread t2 = new Thread(producer2);
		t2.start();

		Thread t3 = new Thread(producer3);
		t3.start();

		Thread consumer = new Thread(consumer1);
		consumer.start();

		// t1 finishes before t2
		t1.join();
		t2.join();
		t3.join();

		consumer.interrupt();
		consumer.join();

		// new Thread(consumer2).start();
		return hm;
	}

	public Map<String, ArrayList<String>> distributeKeys(
			Map<String, ArrayList<String>> keys_umx) {
		Map<String, ArrayList<String>> hm = new HashMap<String, ArrayList<String>>();

		int n_computers = this.computers.length;

		int distribution = 0;
		Set<String> keySet = keys_umx.keySet();
		Iterator<String> itr = keySet.iterator();
		while (itr.hasNext()) {
			String key = itr.next();
			int comp = distribution % n_computers;
			// System.out.println("Computer n°" + comp);
			if (!hm.keySet().contains(this.computers[comp])) {
				ArrayList<String> list = new ArrayList<String>();
				hm.put(this.computers[comp], list);
			}
			hm.get(this.computers[comp]).add(key);
			hm.put(this.computers[comp], hm.get(this.computers[comp]));
			distribution += 1;
		}
		return hm;
	}

	public void shuffle(Map<String, String> umx_machines,
			Map<String, ArrayList<String>> keys_umx) throws IOException {

		List<ProcessBuilder> pb_scp = new ArrayList<ProcessBuilder>();

		Map<String, ArrayList<String>> keysDistribution = this
				.distributeKeys(keys_umx);

		Iterator<String> distribution = keysDistribution.keySet().iterator();

		while (distribution.hasNext()) {
			String computer = distribution.next(); // C133-10, C133-12 ...

			Iterator<String> itr_comp = keysDistribution.get(computer)
					.iterator();

			while (itr_comp.hasNext()) {
				String word = itr_comp.next(); // Car, Beer...
				ArrayList<String> umx_word = keys_umx.get(word); // UM1, UM2 ...

				Iterator<String> itr_umx = umx_word.iterator();
				while (itr_umx.hasNext()) {
					String umx = itr_umx.next();
					String machine = umx_machines.get(umx);
					// System.out.println("Key : " + word + " | Computer src :"
					// + machine + " | Computer dst : " + computer);
					ProcessBuilder pb_slave = new ProcessBuilder("scp",
							"tathan@" + machine + ":/tmp/tathan/maps/" + umx
									+ ".txt", "tathan@" + computer
									+ ":/tmp/tathan/reduces/" + umx + ".txt");

					pb_scp.add(pb_slave);

				}
			}
		}

		Iterator<ProcessBuilder> itr_pb = pb_scp.iterator();
		while (itr_pb.hasNext()) {
			ProcessBuilder pbToLaunch = itr_pb.next();
			this.startThreads(pbToLaunch);
		}

	}

	public void startReducingSortedMaps(
			Map<String, ArrayList<String>> keys_umx,
			Map<String, ArrayList<String>> distributedKeys) throws IOException {

		int count = 0;

		Iterator<String> itr_slave = distributedKeys.keySet().iterator();
		while (itr_slave.hasNext()) {
			String slave = itr_slave.next();

			ArrayList<String> keys = distributedKeys.get(slave);

			for (int i = 0; i < keys.size(); i++) {

				String key = keys.get(i);

				ArrayList<String> umx = keys_umx.get(key);

				System.out.println(key + " : " + umx.toString() + " => " + slave);

				String[] commands = new String[10 + umx.size()];
				commands[0] = "ssh";
				commands[1] = "-o";
				commands[2] = "StrictHostKeyChecking=no";
				commands[3] = "tathan@" + slave;
				commands[4] = "java";
				commands[5] = "-jar";
				commands[6] = "/tmp/tathan/slave.jar";
				commands[7] = "1";
				commands[8] = key;
				commands[9] = "/tmp/tathan/reduces/SM" + count + ".txt";

				
				for (int j=0; j<umx.size(); j++){ 
					commands[10 + j] = "/tmp/tathan/reduces/" + umx.get(j) + ".txt";  
				}
				 

				ProcessBuilder pb = new ProcessBuilder(commands);

				/*ProcessBuilder pb = new ProcessBuilder("ssh", "-o",
						"StrictHostKeyChecking=no", "tathan@" + slave, "java",
						"-jar", "/tmp/tathan/slave.jar", "1", key,
						"/tmp/tathan/reduces/SM" + count + ".txt",
						"/tmp/tathan/reduces/RM" + count + ".txt");*/
				this.startThreads(pb);
			}
			count++;
		}

	}

	public static void main(String[] args) throws IOException,
			InterruptedException {
		MASTER ms = new MASTER("file_dispos.txt");

		// ms.pasteSplits();

		Map<String, String> umx_machines = ms.getMapsComputer();

		ms.buildUMs();

		Map<String, ArrayList<String>> keys_umx = ms.waitForMap();
		System.out.println("Keys / UMx : " + keys_umx.toString());
		System.out.println("UMX / Machines : " + umx_machines.toString());
		ms.shuffle(umx_machines, keys_umx);

		Map<String, ArrayList<String>> distributedKeys = ms
				.distributeKeys(keys_umx);
		System.out.println("Distributed keys : " + distributedKeys.toString());

		ms.startReducingSortedMaps(keys_umx, distributedKeys);

	}

}
