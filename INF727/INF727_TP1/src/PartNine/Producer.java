package PartNine;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.BlockingQueue;

public class Producer implements Runnable {

	private BlockingQueue<Map<String, ArrayList<String>>> queue;
	private Map<String, ArrayList<String>> hm = new HashMap<String, ArrayList<String>>();

	private String computer;
	private int count;

	public Producer(BlockingQueue<Map<String, ArrayList<String>>> queue,
			Map<String, ArrayList<String>> hm, String computer, int count)
			throws InterruptedException {
		this.queue = queue;
		this.hm = hm;
		this.computer = computer;
		this.count = count;
	}
	
	@Override
	public void run() {
		try {
			
			synchronized (this)
            {
				FileReader in = new FileReader("file_dispos.txt");
				BufferedReader br = new BufferedReader(in);
	
				String line;
				// int count = 0;
	
				/*
				 * String[] computers = new String[3]; while ((count < 3) && (line =
				 * br.readLine()) != null) { computers[count] = line ; count ++ ; }
				 */
				System.out.println("START PRODUCER");
				/* GO TO COMPUTER */
				System.out.println("	COMPUTER " + this.computer);
	
				/* RUN SLAVE. JAR */
				System.out.println("		| Run slave jar on file s" + this.count
						+ ".txt with computer " + this.computer);
				ProcessBuilder pb_slave = new ProcessBuilder("ssh",  "-o",
						"StrictHostKeyChecking=no", "tathan@"
						+ this.computer, "java", "-jar", "/tmp/tathan/slave.jar",
						"0", "/tmp/tathan/splits/s" + count + ".txt");
	
				Process p = pb_slave.start();
	
				BufferedReader reader = new BufferedReader(new InputStreamReader(
						p.getInputStream()));
				
				BufferedReader reader_err = new BufferedReader(new InputStreamReader(
						p.getErrorStream()));

				MyThread input_err = new MyThread(reader_err);
				input_err.start();
	
				Keys keys_um = new Keys(reader, hm, "UM" + this.count);
				keys_um.start();
	
				while ((line = reader.readLine()) != null) {
					if(!line.equals("")){
						if (!hm.containsKey(line)) {
							ArrayList<String> list = new ArrayList<String>();
							hm.put(line, list);
						} 
						ArrayList<String> toUpdate = hm.get(line);
						toUpdate.add("UM" + this.count);
						hm.put(line, toUpdate);
					}
					
				}
				// Don't know exactly why this happens
				hm.remove("");
				
				this.queue.put(hm);
				
				br.close();
	
	            
				/*Map<String, ArrayList<String>> end_hm = new HashMap<String,
				ArrayList<String>>(); ArrayList<String> list_null = new
				ArrayList<String>(); end_hm.put("Phase de MAP terminée",list_null); 
				this.queue.put(end_hm);*/
				
				// notifies the consumer thread that
	            // now it can start consuming
	            notify();
				 
				//System.out.println("		|" + this.queue.toString());
            }
		} catch (InterruptedException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
