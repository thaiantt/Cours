package PartNine;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * Create new tathan dir in /tmp and add splits dir and text files
 * 
 * @author thaianthantrong
 *
 */
public class RestartSetup {

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

	public static void main(String[] args) throws IOException {
		RestartSetup st = new RestartSetup();
		String fileName = "file_dispos.txt";

		FileReader in = new FileReader(fileName);
		BufferedReader br = new BufferedReader(in);

		String line;
		int count = 0;
		while ((count < 3) && (line = br.readLine()) != null) {
			System.out.println(line);
			ProcessBuilder pb_ssh = new ProcessBuilder("ssh", "-o",
					"StrictHostKeyChecking=no", "tathan@" + line);
			st.startThreads(pb_ssh);
			
			
			// CREATE TATHAN DIRECTORY
			ProcessBuilder pb_tathan = new ProcessBuilder("mkdir", "-p",
					"/tmp/tathan/");
			st.startThreads(pb_tathan);

			
			// CREATE SPLITS DIRECTORY
			ProcessBuilder pb_mkdir = new ProcessBuilder("ssh", "-o",
					"StrictHostKeyChecking=no", "tathan@"
					+ line, "mkdir", "-p", "/tmp/tathan/splits");
			st.startThreads(pb_mkdir);
			
			
			// CREATE REDUCES DIRECTORY
			ProcessBuilder pb_reduces= new ProcessBuilder("ssh", "-o",
					"StrictHostKeyChecking=no", "tathan@"
					+ line, "mkdir", "-p", "/tmp/tathan/reduces");
			st.startThreads(pb_reduces);
		}

		br.close();
	}

}
