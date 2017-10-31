package PartNine;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Iterator;

class MyThread extends Thread {
	BufferedReader reader;
	String result;

	public MyThread(BufferedReader reader) {
		this.reader = reader;
	}

	public String getResult() {
		return this.result;
	}

	public void run() {
		String line = null;
		StringBuilder builder = new StringBuilder();
		try {
			while ((line = this.reader.readLine()) != null) {
				builder.append(line);
				builder.append(System.getProperty("line.separator"));
			}
			String result = builder.toString();
			System.out.println(result);
			this.result = result;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}

class Keys extends Thread {
	BufferedReader reader;
	Map<String, ArrayList<String>> hm;
	String um;

	public Keys(BufferedReader reader, Map<String, ArrayList<String>> hm,
			String um) {
		this.reader = reader;
		this.hm = hm;
		this.um = um;
	}

	public Map<String, ArrayList<String>> getMap() {
		return this.hm;
	}
	
	public String toString(){
		String s = "";
		Iterator<String> itr = this.hm.keySet().iterator();
		
		while(itr.hasNext()){
			String key = itr.next();
			s += key + "< ";
			Iterator<String> itr_val = this.hm.get(key).iterator();
			while(itr_val.hasNext()){
				String val = itr_val.next();
				if (itr_val.hasNext()){
					s += val + ", ";
				}
				else {
					s += val;
				}
			}
			s += " > \n";
		}
		return s;
	}


	public void run() {
		String line = null;
		try {
			while ((line = this.reader.readLine()) != null) {
				if (!hm.containsKey(line)) {
					ArrayList<String> list = new ArrayList<String>();
					list.add(this.um);
					this.hm.put(line, list);
				} else {
					ArrayList<String> toUpdate = hm.get(line);
					toUpdate.add(this.um);
					this.hm.put(line, toUpdate);
				}
			}
			//System.out.println(this.hm.toString());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}

public class MASTER50 {
	
	public String toString(Map<String, ArrayList<String>> hm){
		String s = "";
		Iterator<String> itr = hm.keySet().iterator();
		
		while(itr.hasNext()){
			String key = itr.next();
			s += key + "< ";
			Iterator<String> itr_val = hm.get(key).iterator();
			while(itr_val.hasNext()){
				String val = itr_val.next();
				if (itr_val.hasNext()){
					s += val + ", ";
				}
				else {
					s += val;
				}
			}
			s += " > \n";
		}
		
		return s;
	}

	public void startThreads(ProcessBuilder pb) throws IOException{
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
	
	public void startKeys(ProcessBuilder pb,
			Map<String, ArrayList<String>> hm, String um) throws IOException {
		Process p = pb.start();

		BufferedReader reader = new BufferedReader(new InputStreamReader(
				p.getInputStream()));

		Keys input = new Keys(reader, hm, um);
		input.start();

	}

	public static void main(String[] args) throws IOException {
		MASTER50 ms = new MASTER50();
		String fileName = "file_dispos.txt";

		FileReader in = new FileReader(fileName);
		BufferedReader br = new BufferedReader(in);

		String line;
		int count = 0;

		/* QUESTION 52 : CREATE HASHMAP TO STORE KEYS : <KEY, <UMx, UMy>> */
		Map<String, ArrayList<String>> hm = new HashMap<String, ArrayList<String>>();

		while ((count < 3) && (line = br.readLine()) != null) {
			// System.out.println("Create file in /tmp/" + line);

			/* GO TO COMPUTER */
			System.out.println("COMPUTER " + line);

			/* CREATE SPLITS DIRECTORY IF NECESSARY */
			System.out.println("	| Create splits directory");
			ProcessBuilder pb_mkdir = new ProcessBuilder("ssh", "tathan@"
					+ line, "mkdir", "-p", "/tmp/tathan/splits/");
			ms.startThreads(pb_mkdir);

			/* RUN SLAVE. JAR */
			System.out.println("	| Run slave jar on file s" + count + ".txt");
			ProcessBuilder pb_slave = new ProcessBuilder("ssh", "tathan@"
					+ line, "java", "-jar", "/tmp/tathan/slave.jar",
					"/tmp/tathan/splits/s" + count + ".txt");
			ms.startKeys(pb_slave, hm, "UM" + count);

			count++;
		}
		br.close();
	}
}
