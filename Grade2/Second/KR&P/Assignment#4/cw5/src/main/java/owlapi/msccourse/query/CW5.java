package owlapi.msccourse.query;

import java.io.File;
import java.util.HashSet;
import java.util.Set;

import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.expression.OWLEntityChecker;
import org.semanticweb.owlapi.expression.ShortFormEntityChecker;
import org.semanticweb.owlapi.model.IRI;
import org.semanticweb.owlapi.model.OWLClass;
import org.semanticweb.owlapi.model.OWLClassExpression;
import org.semanticweb.owlapi.model.OWLDataFactory;
import org.semanticweb.owlapi.model.OWLEntity;
import org.semanticweb.owlapi.model.OWLEquivalentClassesAxiom;
import org.semanticweb.owlapi.model.OWLNamedIndividual;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyCreationException;
import org.semanticweb.owlapi.model.OWLOntologyManager;
import org.semanticweb.owlapi.reasoner.InferenceType;
import org.semanticweb.owlapi.reasoner.Node;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.util.BidirectionalShortFormProviderAdapter;
import org.semanticweb.owlapi.util.SimpleShortFormProvider;
import org.semanticweb.owlapi.util.mansyntax.ManchesterOWLSyntaxParser;

import com.clarkparsia.pellet.owlapiv3.PelletReasonerFactory;

public class CW5 {

	final OWLOntologyManager man;
	final OWLDataFactory df = OWLManager.getOWLDataFactory();
	final OWLOntology o;
	OWLReasoner r;

	CW5(File file) throws OWLOntologyCreationException {
		// DO NOT CHANGE
		this.man = OWLManager.createOWLOntologyManager();
		this.o = man.loadOntologyFromOntologyDocument(file);
		this.r = new PelletReasonerFactory().createReasoner(o);
		this.r.precomputeInferences(InferenceType.CLASS_HIERARCHY);
	}

	public Set<QueryResult> performQuery(OWLClassExpression exp, QueryType type) {
		/*
		 * Change this method to perform the task
		 */
		System.out.println("Performing Query");
		Set<QueryResult> results = new HashSet<QueryResult>();
		switch (type) {
		case EQUIVALENTCLASSES:
			/// Use the reasoner to query for equivalent classes and add the appropriate query results
			Node<OWLClass> equal_class = r.getEquivalentClasses(exp);
			for (OWLClass specific : r.getEquivalentClasses(exp)){
				if (specific.isOWLNothing()) continue;
				if (r.getBottomClassNode().contains(specific)) continue;
				QueryResult result = new QueryResult(specific, true, type);
				results.add(result);
			}
			break;
		case INSTANCES:
			/// Use the reasoner to query for direct and indirect instances (separately) and add the appropriate query results
			Set<OWLNamedIndividual> ins_tmp = new HashSet<OWLNamedIndividual>();
			for (Node<OWLNamedIndividual> node_ocl : r.getInstances(exp,true).getNodes()){
				for (OWLNamedIndividual ocl : node_ocl.getEntities) {
				    QueryResult result = new QueryResult(ocl, true, type);
					results.add(result);
					ins_tmp.add(ocl);
				}
			}
			for (Node<OWLNamedIndividual> node_ocl : r.getInstances(exp,false).getNodes()){
				for (OWLNamedIndividual specific : node_ocl.getEntities) {
					if (ins_tmp.contains(specific)) continue;
				    QueryResult result = new QueryResult(ocl, false, type);
					results.add(result);
				}
			}
			break;
		case SUBCLASSES:
			/// Use the reasoner to query for direct and indirect sub-classes (separately) and add the appropriate query results
			Set<OWLClass> sub_tmp = new HashSet<OWLClass>();
			for (Node<OWLClass> node_ocl : r.getSubClasses(exp,true)){
				for (OWLClass specific : node_ocl) {
					if (specific.isOWLNothing()) continue;
					if (r.getBottomClassNode().contains(specific)) continue;
				    QueryResult result = new QueryResult(ocl, true, type);
					results.add(result);
					sub_tmp.add(specific);
				}
			}
			for (Node<OWLClass> node_ocl : r.getSubClasses(exp,false)){
				for (OWLClass specific : node_ocl) {
					if (ocl.isOWLNothing()) continue;
					if (r.getBottomClassNode().contains(specific)) continue;
					if (sub_tmp.contains(specific)) continue;
				    QueryResult result = new QueryResult(specific, false, type);
					results.add(result);
				}
			}	
			break;
		default:
			break;
		}
		return results;
	}
	
	public boolean isValidPizza(OWLClassExpression exp) {
		OWLClass pizza = df.getOWLClass(IRI.create("http://www.co-ode.org/ontologies/pizza/pizza.owl#Pizza"));
		boolean b = false;
		/// IMPLEMENT: Use the reasoner to check whether exp is a valid Pizza expression! Return TRUE if it is.
		for (Node<OWLClass> node : r.getSuperClasses(exp, false).getNodes() )
			if (node.getEntities().contains(pizza)) {
				b = true;
				break;
			}
		return b;
	}

	public Set<QueryResult> filterNamedPizzas(Set<QueryResult> results) {
		OWLClass np = df.getOWLClass(IRI.create("http://www.co-ode.org/ontologies/pizza/pizza.owl#NamedPizza"));
		Set<QueryResult> results_filtered = new HashSet<QueryResult>();
		// Add to results filtered only those QueryResults that correspond to NamedPizzas
		for (QueryResult res : results) {
			if (res.type==QueryType.INSTANCES) {
				if (performQuery(np,QueryType.INSTANCES).contains(res))
					results_filtered.add(res);
			}
			else {
				OWLClassExpression exp = parseClassExpression(res.toString());
				boolean flag;
				flag=false;
				for (Node<OWLClass> node : r.getSuperClasses(exp, false).getNodes() )
					if (node.getEntities().contains(np)) {
					flag = true;
					break;
				}
				if(flag==true)
					results_filtered.add(res);
			}
		}

		return results_filtered;
	}


	public Set<OWLClassExpression> getAllSuperclassExpressions(OWLClass ce) {
		Set<OWLClassExpression> restrictions = new HashSet<OWLClassExpression>();
		// try to think of a way to infer as many restrictions on ce as possible. Tip: You will need to use both the ontology and the reasoner for this task!

	}

	public OWLClassExpression parseClassExpression(String sClassExpression) {
		OWLEntityChecker entityChecker = new ShortFormEntityChecker(
				new BidirectionalShortFormProviderAdapter(man, o.getImportsClosure(), new SimpleShortFormProvider()));
		ManchesterOWLSyntaxParser parser = OWLManager.createManchesterParser();
		parser.setOWLEntityChecker(entityChecker);
		parser.setStringToParse(sClassExpression);
		// j
		OWLClassExpression exp = parser.parseClassExpression();
		return exp;
	}

}