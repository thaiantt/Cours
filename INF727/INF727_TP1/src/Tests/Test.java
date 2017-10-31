package Tests;
public class Test {

	/* See the application thread : the name is main */
	/*
	 * public static void main(String[] args) {
	 * System.out.println("Le nom du thread principal est " +
	 * Thread.currentThread().getName()); }
	 */

	/* New code */

	public static void main(String[] args) {
		TestThread t = new TestThread("A");
		TestThread t2 = new TestThread("  B", t);

		try {
		      Thread.sleep(1000);
		    } catch (InterruptedException e) {
		      e.printStackTrace();
		    }
		
			// See that Threads don't run at the same time
		    System.out.println("statut du thread " + t.getName() + " = " + t.getState());
		    System.out.println("statut du thread " + t2.getName() + " = " +t2.getState());        

	}

}
