import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.TreeMap;

import com.google.gson.Gson;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;

// Usage: java ProcessData <inputFile> <outputFile>

// Processes KBP data piped from process_kbp_data.py. Outputs to stdout the subtree of parse tree containing the 
public class ProcessData {

	static Properties props;
	static StanfordCoreNLP pipeline;
	static Gson gson = new Gson();

	public static void main(String[] args) throws IOException {
		// set up corenlp stuff
		props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, parse");
		props.setProperty("parse.binaryTrees", "true");
		pipeline = new StanfordCoreNLP(props);

		String inputFilename = args[0];
		String outputFilename = args[1];

		//processInput();
		processInputFromFile(inputFilename, outputFilename);
	}

	// Read/write from/to files.
	public static void processInputFromFile(String inputFilename, String outputFilename) throws IOException {
		BufferedReader in = new BufferedReader(new FileReader(inputFilename));
		PrintWriter writer = new PrintWriter(outputFilename, "UTF-8");
		String line;
		while ((line = in.readLine()) != null) {
			String result = processLine(line);

			if (result != null) {
				// write to file
				writer.println(result);
			}
		}
		in.close();
		writer.close();
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
	public static String processLine(String jsonLine) {
		//System.err.println(jsonLine);
		String[] vals = gson.fromJson(jsonLine, String[].class);

		// [new_gloss, subject_entity, object_entity, str(subject_begin), str(subject_end), str(object_begin), str(object_end), relations]
		String gloss = vals[0];
		String subject_entity = vals[1];
		String object_entity = vals[2];
		int subject_begin = Integer.parseInt(vals[3]);
		int subject_end = Integer.parseInt(vals[4]);
		int object_begin = Integer.parseInt(vals[5]);
		int object_end = Integer.parseInt(vals[6]);
		String rels = vals[7];

		if (subject_begin == object_begin || subject_begin < 0 || subject_end < 0 || object_begin < 0 || object_end < 0) {
			return null;
		}

		// get the parse for the desired subtree
		String parse = getParse(gloss, subject_entity, object_entity, subject_begin, subject_end, object_begin, object_end);

		// parse is null if we have more than a single sentence on a given line
		if (parse != null) {
			String output = makeOutput(parse, rels);
			//System.out.println(output);
			return output;
		}
		else {
			return null;
		}
	}

	public static String makeOutput(String parse, String relations) {
		return parse + "\t" + relations;
	}

	// Returns a string representing the subtree of the parse tree representing the lowest common ancestor of the 2 mentions.
	public static String getParse(String text, String subject_entity, String object_entity, int subject_begin, int subject_end, int object_begin, int object_end) {
		Annotation document = new Annotation(text);
		pipeline.annotate(document);
		List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

		// we should never have more than a single sentence per line in the corpus
		if (sentences.size() > 1) {
			return null;
		}

		CoreMap sentence = sentences.get(0);
		Tree binarizedTree = sentence.get(TreeCoreAnnotations.BinarizedTreeAnnotation.class); // the root
		List<Tree> leaves = binarizedTree.getLeaves();

		// indices in our data set (mention locations in the sentence) can be also indices into Tree.getLeaves()

		// we define m1 to be the subject, and m2 to be the object

		// get all the m1 and m2 nodes
		// treemaps so that the indices are in sorted order
		Map<Integer, Tree> m1Nodes = new TreeMap<Integer, Tree>();
		Map<Integer, Tree> m2Nodes = new TreeMap<Integer, Tree>();

		int m1_start = Math.min(subject_begin, subject_end);
		int m1_end = Math.max(subject_begin, subject_end);
		int m2_start = Math.min(object_begin, object_end);
		int m2_end = Math.max(object_begin, object_end);

		for (int i = 0; i < leaves.size(); i++) {
			Tree currLeaf = leaves.get(i);
			if (i >= m1_start && i < m1_end) m1Nodes.put(i, currLeaf);
			if (i >= m2_start && i < m2_end) m2Nodes.put(i, currLeaf);
		}

		// get the head node for m1 and m2 (highest node in the parse tree - lowest distance to root)
		Tree m1Head = null;
		Tree m2Head = null;
		int m1HeadIndex = -1;
		int m2HeadIndex = -1;
		int m1MinDist = 10000;
		int m2MinDist = 10000;

		for (Map.Entry<Integer, Tree> entry : m1Nodes.entrySet()) {
			int index = entry.getKey();
			Tree leaf = entry.getValue();

			int distToRoot = leaf.depth(binarizedTree);
			if (distToRoot < m1MinDist) {
				m1MinDist = distToRoot;
				m1Head = leaf;
				m1HeadIndex = index;
			}
		}

		for (Map.Entry<Integer, Tree> entry : m2Nodes.entrySet()) {
			int index = entry.getKey();
			Tree leaf = entry.getValue();

			int distToRoot = leaf.depth(binarizedTree);
			if (distToRoot < m2MinDist) {
				m2MinDist = distToRoot;
				m2Head = leaf;
				m2HeadIndex = index;
			}
		}

		// remove the non-head words from the tree
		for (Map.Entry<Integer, Tree> entry : m1Nodes.entrySet()) {
			int index = entry.getKey();
			if (index == m1HeadIndex) continue;

			// this is the node we want to remove
			Tree leaf = entry.getValue(); // just the word
			Tree parent = leaf.parent(binarizedTree); // (POS word) - this is what we want to remove
			Tree parent2 = parent.parent(binarizedTree); // the parent of (POS word)

			// parent2 is the parent of parent, and parent is the node we want to remove
			removeNode(parent2, parent);

			// if parent2 now has no children, we need to remove parent2 also (so get his parent and remove parent2)
			Tree parentAbove = parent2;

			// keep going up the tree and removing the parent if he no longer has children
			while (parentAbove.children().length == 0) {
				Tree parentAboveAbove = parentAbove.parent(binarizedTree);
				removeNode(parentAboveAbove, parentAbove);
				parentAbove = parentAboveAbove; // move up the tree
			}
		}
		for (Map.Entry<Integer, Tree> entry : m2Nodes.entrySet()) {
			int index = entry.getKey();
			if (index == m2HeadIndex) continue;

			// this is the node we want to remove
			Tree leaf = entry.getValue(); // just the word
			Tree parent = leaf.parent(binarizedTree); // (POS word) - this is what we want to remove
			Tree parent2 = parent.parent(binarizedTree); // the parent of (POS word)

			// parent2 is the parent of parent, and parent is the node we want to remove
			removeNode(parent2, parent);

			// if parent2 now has no children, we need to remove parent2 also (so get his parent and remove parent2)
			Tree parentAbove = parent2;

			// keep going up the tree and removing the parent if he no longer has children
			while (parentAbove.children().length == 0) {
				Tree parentAboveAbove = parentAbove.parent(binarizedTree);
				removeNode(parentAboveAbove, parentAbove);
				parentAbove = parentAboveAbove; // move up the tree
			}
		}

		// replace the head node words with entity strings		
		m1Head.setValue(subject_entity);
		m2Head.setValue(object_entity);

		// now binarizedTree has the correct leaves (the mentions replaced by entity strings)

		// now get the LCA of the 2 heads
		// there is only a single path between them, which is through their LCA.
		// to get the LCA we need to find the highest node on the path, though (the one closest to the root)

		List<Tree> path = binarizedTree.pathNodeToNode(m1Head, m2Head);

		// the highest node on the path through the dependency tree between the 2 mentions (this path has to go through the LCA)
		Tree lca = null;

		// the distance of the closest node to the root through which the path goes
		// (which is not a node in either of the mentions!)
		int maxDepth = -1;

		for (Tree node : path) {
			// skip this node if it's either of the heads
			if (node == m1Head || node == m2Head) continue;

			int depth = node.depth();
			if (depth > maxDepth) {
				maxDepth = depth;
				lca = node;
			}
		}

		return lca.toString();
	}

	// Removes child_to_remove from parent.
	public static void removeNode(Tree parent, Tree child_to_remove) {
		// get the index of child_to_remove in parent's children list
		int ind = -1;
		for (int j = 0; j < parent.children().length; j++) {
			Tree curr_child = parent.children()[j];
			if (curr_child == child_to_remove) {
				ind = j;
			}
		}
		parent.removeChild(ind);
	}
}