package PartFive;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class DEPLOY {
	
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

	public static void main(String[] args) throws IOException {
		DEPLOY dp = new DEPLOY();
		String fileName = "file_dispos.txt";

		FileReader in = new FileReader(fileName);
		BufferedReader br = new BufferedReader(in);

		String line;
		while ((line = br.readLine()) != null) {
			System.out.println("Create file in /tmp/" + line);

			// Create new dir
			/*ProcessBuilder pb_mkdir = new ProcessBuilder("ssh", "tathan@"
					+ line, "mkdir", "-p", "/tmp/tathan/");
			dp.startThreads(pb_mkdir);*/

			// Copy slave.jar
			ProcessBuilder pb_scp = new ProcessBuilder(
					"scp",
					"/Users/thaianthantrong/Documents/MS_BIG_DATA/Cours/INF727/slave.jar",
					"tathan@" + line + ":/tmp/tathan/slave.jar");
			dp.startThreads(pb_scp);

		}
	}

}
