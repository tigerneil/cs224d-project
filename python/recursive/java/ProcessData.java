import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import com.google.gson.Gson;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;

// Processes KBP data piped from process_kbp_data.py. Outputs to stdout the subtree of parse tree containing the 
public class ProcessData {

	static Properties props;
	static StanfordCoreNLP pipeline;
	static Gson gson = new Gson();

	public static void main(String[] args) throws IOException {
		// the code around the call to processInput() just suppresses corenlp output so that we can pipe
		// the output of this file through

		// this is your print stream, store the reference
		PrintStream err = System.err;

		// now make all writes to the System.err stream silent 
		System.setErr(new PrintStream(new OutputStream() {
			public void write(int b) {
			}
		}));

		// set up corenlp stuff
		props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, parse");
		props.setProperty("parse.binaryTrees", "true");
		pipeline = new StanfordCoreNLP(props);

		processInput();

		// set everything back to its original state afterwards
		System.setErr(err); 
	}

	// Processes input from stdin, and calls processLine on each line.
	public static void processInput() throws IOException {
		// Wrap the System.in inside BufferedReader
		// But do not close it in a finally block, as we 
		// did no open System.in; enforcing the rule that
		// he who opens it, closes it; leave the closing to the OS.
		BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
		String line;
		while ((line = in.readLine()) != null) {
			processLine(line);
		}
	}

	// Processes each line of the python-processed KBP data. (needs to be the second step in that pipeline)
	public static void processLine(String jsonLine) {
		String[] vals = gson.fromJson(jsonLine, String[].class);

		// [new_gloss, str(ind1), str(ind2), relations]
		//System.out.println(Arrays.toString(vals));

		String gloss = vals[0];
		int m1_ind = Integer.parseInt(vals[1]);
		int m2_ind = Integer.parseInt(vals[2]);
		String rels = vals[3];

		// get the parse for the desired subtree
		String parse = getParse(gloss, m1_ind, m2_ind);

		String output = makeOutput(parse, rels);
		System.out.println(output);
	}

	public static String makeOutput(String parse, String relations) {
		return parse + "\t" + relations;
	}

	// Returns a string representing the subtree of the parse tree representing the lowest common ancestor of the 2 mentions.
	// Assume ind1 and ind2 are sorted in increasing order by the caller.
	public static String getParse(String text, int ind1, int ind2) {
		// create an empty Annotation just with the given text
		Annotation document = new Annotation(text);

		// run all Annotators on this text
		pipeline.annotate(document);

		// these are all the sentences in this document
		// a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
		List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

		// we should never have more than a single sentence per line in the corpus
		assert(sentences.size() <= 1);

		CoreMap sentence = sentences.get(0);
		Tree binarizedTree = sentence.get(TreeCoreAnnotations.BinarizedTreeAnnotation.class);
		List<Tree> leaves = binarizedTree.getLeaves();

		// indices in our data set (mention locations in the sentence) can be also indices into Tree.getLeaves()

		int i = 0;
		Tree t1 = null;
		Tree t2 = null;
		for (Tree leaf : leaves) {
			if (i == ind1) t1 = leaf;
			if (i == ind2) t2 = leaf;
			i++;
		}

		List<Tree> path = binarizedTree.pathNodeToNode(t1, t2);

		// the lowest node on the path through the dependency tree between the 2 mentions (this path has to go through the LCA)
		Tree lca = null;

		// the height of the highest node through which the path goes
		int maxHt = -1;

		for (Tree node : path) {
			if (node.depth() > maxHt) {
				maxHt = node.depth();
				lca = node;
			}
		}

		// print the leaves on the path
//		for (Tree leaf : lca.getLeaves()) {
//			System.out.println(leaf.nodeString());
//		}
		
		return lca.toString();
	}
}