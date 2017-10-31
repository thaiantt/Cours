package PartNine;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SLAVE {

	private static final String PATH_TO_FILES = "/tmp/tathan/";

	/* QUESTION 54 : MODE */
	public String mode;

	public String key;

	/* Absolute path to sx.txt */
	public String[] filename;

	public SLAVE(String mode, String[] filename) {
		this.mode = mode;
		if (this.mode.equals("1")) {
			int nb_files = filename.length - 1;

			String[] files = new String[nb_files];
			for (int i = 0; i < nb_files; i++) {
				files[i] = filename[i + 1];
			}

			this.key = filename[0];
			this.filename = files;
		} else {
			this.filename = filename;
		}
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

	/*
	 * Used for mode == 0
	 */
	public void createUM() throws IOException {

		ProcessBuilder pb_mkdir = new ProcessBuilder("mkdir", "-p",
				"/tmp/tathan/maps/");
		this.startThreads(pb_mkdir);

		FileReader in = null;
		try {
			in = new FileReader(this.filename[0]);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		BufferedReader br = new BufferedReader(in);

		String parts = this.filename[0].substring(
				this.filename[0].length() - 5, this.filename[0].length() - 4);

		PrintWriter writer = new PrintWriter(PATH_TO_FILES + "maps/UM" + parts
				+ ".txt", "UTF-8");

		String line;

		while ((line = br.readLine()) != null) {
			// System.out.println(line);

			// Remove special characters
			String line_adjusted = line.replaceAll("[^\\x00-\\x7f]+", "");

			String[] words = line_adjusted.split(" ");

			for (int i = 0; i < words.length; i++) {
				if (!words[i].equals("")){
					writer.println(words[i] + "," + 1);
					
					System.out.println(words[i]);
				}
			}
		}
		in.close();

		writer.close();
	}

	/*
	 * Used for mode == 1
	 */
	public void createSM(String smx) throws IOException {
		PrintWriter writer = new PrintWriter(smx, "UTF-8");

		for (int i = 0; i < this.filename.length; i++) {
			FileReader in = new FileReader(this.filename[i]);
			BufferedReader br = new BufferedReader(in);

			String line;

			while ((line = br.readLine()) != null) {

				String[] parts = line.split(",");
				String key_um = parts[0];

				if (key_um.equals(this.key)
						|| key_um.toLowerCase().equals(this.key)) {
					// System.out.println(line);
					writer.println(line);
				}

			}
			br.close();
		}

		writer.close();

	}

	/*
	 * Used for mode == 1
	 */
	public void createRM(String smx) throws IOException {
		FileReader in = new FileReader(smx);
		String s = "";
		Pattern p = Pattern.compile("SM\\d.txt");
		Matcher m = p.matcher(smx);
		while (m.find()) {
		    s = Character.toString(m.group(0).charAt(2));
		    // s now contains "BAR"
		}
		
		String rmx = PATH_TO_FILES + "/reduces/RM" + s + ".txt";
				
		PrintWriter writer = new PrintWriter(rmx, "UTF-8");
		BufferedReader br = new BufferedReader(in);

		String line;

		int count = 0 ;
		String key = "";
		while ((line = br.readLine()) != null) {
			key = line.split(",")[0];
			count++;
		}

		br.close();
		
		writer.println(key + " " + count);
		writer.close();

	}

	public void launchSlave(String smx) throws IOException {
		if (this.mode.equals("0")) {
			this.createUM();
		} else {
			this.createSM(smx);
			this.createRM(smx);
		}
	}

	/*
	 * params mode : String 0 (compute Sx -> UMx) or 1 (compute UMx -> SMx) file
	 * : String if mode == 0 : file Sx if mode == 1 : file UMx
	 */
	public static void main(String[] args) throws IOException {
		//System.out.println("Launching Slave in mode : " + args[0]);
		int nb_files = args.length - 1;

		String[] files = new String[nb_files];
		for (int i = 0; i < nb_files; i++) {
			files[i] = args[i + 1];
		}
		SLAVE slave = new SLAVE(args[0], files);

		// For mode = 1
		if (args.length > 2){
			String smx = args[2];
			slave.launchSlave(smx);
		}
		else{
			slave.launchSlave("0");
		}
	}
}