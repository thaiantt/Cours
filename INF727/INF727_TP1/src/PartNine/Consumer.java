package PartNine;

import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.BlockingQueue;

public class Consumer implements Runnable {
	private BlockingQueue<Map<String, ArrayList<String>>> queue;

	public Consumer(BlockingQueue<Map<String, ArrayList<String>>> queue) {
		this.queue = queue;
	}

	@Override
	public void run() {
		// TODO Auto-generated method stub
		System.out.println("START CONSUMER");
		Map<String, ArrayList<String>> s;
		// s = queue.poll(this.time, TimeUnit.SECONDS);
		try {
			
			s = queue.poll();
			//System.out.println("--- CONSUMER : Print results");
			while (s != null) {
				if (s.containsKey("Phase de MAP terminée")) {
					System.out.println("-----------------------------------");
					System.out.println("------ Phase de MAP terminée ------");
					System.out.println("-----------------------------------");
				} else {
					System.out.println("		| Obtained with CONSUMER : " + s);
				}
				s = queue.take();
				
			}
			// Wake up producer thread
            //notify();
			
			System.out.println("--- PHASE DE MAP TERMINEE ---");
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
