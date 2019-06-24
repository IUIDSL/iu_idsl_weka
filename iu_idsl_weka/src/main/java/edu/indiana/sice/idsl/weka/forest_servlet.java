package edu.indiana.sice.idsl.weka;

import java.io.*;
import java.net.*; //URLEncoder,InetAddress
import java.text.*;
import java.util.*;
import java.util.regex.*;
import javax.servlet.*;
import javax.servlet.http.*;

import com.oreilly.servlet.*; //MultipartRequest,Base64Encoder,Base64Decoder
import com.oreilly.servlet.multipart.DefaultFileRenamePolicy;

import chemaxon.formats.*;
import chemaxon.marvin.io.MolExportException;
import chemaxon.struc.Molecule;
import chemaxon.marvin.calculations.MarkushEnumerationPlugin;
import chemaxon.marvin.plugin.PluginException;

import weka.core.*;
import weka.core.converters.*; //ArffLoader, CSVLoader, ArffSaver
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;

import edu.indiana.sice.idsl.util.http.*;

/// To do:
///	[ ] Input formats include CSV (delim control) with additional (explanatory) variables
///	[ ] Input formats include SDF with additional variables
///	[ ] For CSV or SDF, control of activity (result) field
///	[ ] Allow random partition of single file for testing/validation.
///	[ ] Improved output visualization.
///	[ ] Threading/progress?
///

/**	Weka-based random forest machine learning web app.

	@author Jeremy J Yang
*/
public class forest_servlet extends HttpServlet
{
  private static String SERVLETNAME=null;
  private static String CONTEXTPATH=null;
  private static String LOGDIR=null;
  private static String APPNAME=null;   // configured in web.xml
  private static String UPLOADDIR=null;	// configured in web.xml
  private static Integer MAX_POST_SIZE=10*1024*1024; // configured in web.xml
  private static String SCRATCHDIR=null; // configured in web.xml
  private static int N_MAX=100;	// configured in web.xml
  private static String PREFIX=null;
  private static int scratch_retire_sec=3600;
  private static MolImporter molReader=null;
  private static ServletContext CONTEXT=null;
  private static ServletConfig CONFIG=null;
  private static ResourceBundle rb=null;
  private static PrintWriter out=null;
  private static ArrayList<String> outputs=null;
  private static ArrayList<String> errors=null;
  private static HttpParams params=null;
  private static int serverport=0;
  private static String SERVERNAME=null;
  private static String REMOTEHOST=null;
  private static String datestr=null;
  private static File logfile=null;
  private static String color1="#EEEEEE";
  private static File fileT=null; // "infile_train"
  private static File fileE=null; // "infile_eval"

  /////////////////////////////////////////////////////////////////////////////
  public void doPost(HttpServletRequest request,HttpServletResponse response)
      throws IOException,ServletException
  {
    serverport=request.getServerPort();
    SERVERNAME=request.getServerName();
    if (SERVERNAME.equals("localhost")) SERVERNAME=InetAddress.getLocalHost().getHostAddress();
    REMOTEHOST=request.getHeader("X-Forwarded-For"); // client (original)
    if (REMOTEHOST!=null)
    {
      String[] addrs=Pattern.compile(",").split(REMOTEHOST);
      if (addrs.length>0) REMOTEHOST=addrs[addrs.length-1];
    }
    else
    {
      REMOTEHOST=request.getRemoteAddr(); // client (may be proxy)
    }

    CONTEXTPATH=request.getContextPath();  //e.g. "/weka"
    rb=ResourceBundle.getBundle("LocalStrings",request.getLocale());

    MultipartRequest mrequest=null;
    if (request.getMethod().equalsIgnoreCase("POST"))
    {
      try { mrequest=new MultipartRequest(request,UPLOADDIR,10*1024*1024,"ISO-8859-1",
                                    new DefaultFileRenamePolicy()); }
      catch (IOException lEx) {
        this.getServletContext().log("not a valid MultipartRequest",lEx); }
    }

    // main logic:
    ArrayList<String> cssincludes = new ArrayList<String>(Arrays.asList("biocomp.css"));
    ArrayList<String> jsincludes = new ArrayList<String>(Arrays.asList("/marvin/marvin.js","biocomp.js","ddtip.js"));
    boolean ok=initialize(request,mrequest);
    if (!ok)
    {
      response.setContentType("text/html");
      out=response.getWriter();
      out.print(HtmUtils.HeaderHtm(SERVLETNAME, jsincludes, cssincludes, JavaScript(), "", color1, request, "tomcat"));
      out.print(HtmUtils.FooterHtm(errors,true));
    }
    else if (mrequest!=null)		//method=POST, normal operation
    {
      if (mrequest.getParameter("forest").equals("TRUE"))
      {
        response.setContentType("text/html");
        out=response.getWriter();
        out.print(HtmUtils.HeaderHtm(SERVLETNAME, jsincludes, cssincludes, JavaScript(), "", color1, request, "tomcat"));
        out.println(FormHtm(mrequest,response));
        Date t_i = new Date();
        Learner(mrequest,response);
        Date t_f = new Date();
        long t_d=t_f.getTime()-t_i.getTime();
        int t_d_min = (int)(t_d/60000L);
        int t_d_sec = (int)((t_d/1000L)%60L);
        errors.add(SERVLETNAME+": elapsed time: "+t_d_min+"m "+t_d_sec+"s");
        out.println(HtmUtils.OutputHtm(outputs));
        out.println(HtmUtils.FooterHtm(errors,true));
        HtmUtils.PurgeScratchDirs(Arrays.asList(SCRATCHDIR),scratch_retire_sec,false,".",(HttpServlet) this);
      }
    }
    else
    {
      String help=request.getParameter("help");	// GET param
      String downloadtxt=request.getParameter("downloadtxt"); // POST param
      String downloadfile=request.getParameter("downloadfile"); // POST param
      if (help!=null)	// GET method, help=TRUE
      {
        response.setContentType("text/html");
        out=response.getWriter();
        out.print(HtmUtils.HeaderHtm(SERVLETNAME, jsincludes, cssincludes, JavaScript(), "", color1, request, "tomcat"));
        out.println(HelpHtm());
        out.println(HtmUtils.FooterHtm(errors,true));
      }
      else if (downloadtxt!=null && downloadtxt.length()>0) // POST param
      {
        ServletOutputStream ostream=response.getOutputStream();
        HtmUtils.DownloadString(response,ostream,downloadtxt,request.getParameter("fname"));
      }
      else if (downloadfile!=null && downloadfile.length()>0) // POST param
      {
        ServletOutputStream ostream=response.getOutputStream();
        HtmUtils.DownloadFile(response,ostream,downloadfile,request.getParameter("fname"));
      }
      else	// GET method, initial invocation of servlet w/ no params
      {
        response.setContentType("text/html");
        out=response.getWriter();
        out.print(HtmUtils.HeaderHtm(SERVLETNAME, jsincludes, cssincludes, JavaScript(), "", color1, request, "tomcat"));
        out.println(FormHtm(mrequest,response));
        out.println("<SCRIPT>go_init(window.document.mainform)</SCRIPT>");
        out.println(HtmUtils.FooterHtm(errors,true));
      }
    }
  }
  /////////////////////////////////////////////////////////////////////////////
  private boolean initialize(HttpServletRequest request,MultipartRequest mrequest)
      throws IOException,ServletException
  {
    SERVLETNAME=this.getServletName();
    outputs = new ArrayList<String>();
    errors = new ArrayList<String>();
    params = new HttpParams();
    Calendar calendar=Calendar.getInstance();

    String logo_htm="<TABLE CELLSPACING=5 CELLPADDING=5><TR><TD>";
    String imghtm=("<IMG HEIGHT=\"60\" BORDER=0 SRC=\"/tomcat"+CONTEXTPATH+"/images/iu_logo.png\">");
    String tiphtm=("Indiana University Cheminformatics &amp; Chemogenomics Research Group");
    String href=("http://iuccrg.wordpress.com");
    logo_htm+=(HtmUtils.HtmTipper(imghtm,tiphtm,href,200,"white"));
    logo_htm+="</TD><TD>";

    imghtm=("<IMG HEIGHT=\"60\" BORDER=0 SRC=\"/tomcat"+CONTEXTPATH+"/images/WEKA_logo.png\">");
    tiphtm=("WEKA from The University of Waikato");
    href=("http://www.cs.waikato.ac.nz/ml/weka/");
    logo_htm+=(HtmUtils.HtmTipper(imghtm,tiphtm,href,200,"white"));
    logo_htm+="</TD><TD>";

    imghtm=("<IMG BORDER=0 SRC=\"/tomcat"+CONTEXTPATH+"/images/chemaxon.png\">");
    tiphtm=("JChem and Marvin from ChemAxon Ltd.");
    href=("http://www.chemaxon.com");
    logo_htm+=(HtmUtils.HtmTipper(imghtm,tiphtm,href,200,"white"));

    logo_htm+="</TD></TR></TABLE>";
    errors.add(logo_htm);

    //Create webapp-specific log dir if necessary:
    File dout=new File(LOGDIR);
    if (!dout.exists())
    {
      boolean ok=dout.mkdir();
      System.err.println("LOGDIR creation "+(ok?"succeeded":"failed")+": "+LOGDIR);
      if (!ok)
      {
        errors.add("ERROR: could not create LOGDIR: "+LOGDIR);
        return false;
      }
    }

    String logpath=LOGDIR+"/"+SERVLETNAME+".log";
    logfile=new File(logpath);
    if (!logfile.exists())
    {
      logfile.createNewFile();
      logfile.setWritable(true,true);
      PrintWriter out_log=new PrintWriter(logfile);
      out_log.println("date\tip\tN"); 
      out_log.flush();
      out_log.close();
    }
    if (!logfile.canWrite())
    {
      errors.add("ERROR: Log file not writable.");
      return false;
    }
    BufferedReader buff=new BufferedReader(new FileReader(logfile));
    if (buff==null)
    {
      errors.add("ERROR: Cannot open log file.");
      return false;
    }

    int n_lines=0;
    String line=null;
    String startdate=null;
    while ((line=buff.readLine())!=null)
    {
      ++n_lines;
      String[] fields=Pattern.compile("\\t").split(line);
      if (n_lines==2) startdate=fields[0];
    }
    if (n_lines>2)
    {
      calendar.set(Integer.parseInt(startdate.substring(0,4)),
               Integer.parseInt(startdate.substring(4,6))-1,
               Integer.parseInt(startdate.substring(6,8)),
               Integer.parseInt(startdate.substring(8,10)),
               Integer.parseInt(startdate.substring(10,12)),0);

      DateFormat df=DateFormat.getDateInstance(DateFormat.FULL,Locale.US);
      errors.add("since "+df.format(calendar.getTime())+", times used: "+(n_lines-1));
    }

    calendar.setTime(new Date());
    datestr=String.format("%04d%02d%02d%02d%02d",
      calendar.get(Calendar.YEAR),
      calendar.get(Calendar.MONTH)+1,
      calendar.get(Calendar.DAY_OF_MONTH),
      calendar.get(Calendar.HOUR_OF_DAY),
      calendar.get(Calendar.MINUTE));

    Random rand = new Random();
    PREFIX=SERVLETNAME+"."+datestr+"."+String.format("%03d",rand.nextInt(1000));

    if (mrequest==null) return true;

    /// Stuff for a run:

    for (Enumeration e=mrequest.getParameterNames(); e.hasMoreElements(); )
    {
      String key=(String)e.nextElement();
      if (mrequest.getParameter(key)!=null)
        params.setVal(key,mrequest.getParameter(key));
    }

    if (params.isChecked("verbose"))
    {
      errors.add("server: "+CONTEXT.getServerInfo()+" [API:"+CONTEXT.getMajorVersion()+"."+CONTEXT.getMinorVersion()+"]");
      errors.add("ServletContextName: "+CONTEXT.getServletContextName());
      errors.add("Weka ver: "+weka.core.Version.VERSION);
      errors.add("JChem ver: "+com.chemaxon.version.VersionInfo.getVersion());
    }

    fileT=mrequest.getFile("infile_train");
    fileE=mrequest.getFile("infile_eval");

    return true;
  }
  /////////////////////////////////////////////////////////////////////////////
  private static String FormHtm(MultipartRequest mrequest,HttpServletResponse response)
	throws IOException
  {
    String htm=
    ("<FORM NAME=\"mainform\" METHOD=\"POST\" ACTION=\""+response.encodeURL(SERVLETNAME)+"\" ENCTYPE=\"multipart/form-data\">\n")
      +("<TABLE WIDTH=\"100%\"><TR><TD><H2>"+APPNAME+"</H2></TD>\n")
      +("<TD>- random forest analysis for molecular classification</TD>\n")
      +("<TD ALIGN=RIGHT>\n")
      +("<BUTTON TYPE=BUTTON onClick=\"void window.open('"+response.encodeURL(SERVLETNAME)+"?help=TRUE','helpwin','width=600,height=400,scrollbars=1,resizable=1')\"><B>Help</B></BUTTON>\n")
      +("<BUTTON TYPE=BUTTON onClick=\"window.location.replace('"+response.encodeURL(SERVLETNAME)+"')\"><B>Reset</B></BUTTON>\n")
      +("</TD></TR></TABLE>\n")
      +("<HR>\n")
      +("<INPUT TYPE=HIDDEN NAME=\"forest\">\n")
      +("<TABLE WIDTH=\"100%\" CELLPADDING=5 CELLSPACING=5>\n")
      +("<TR BGCOLOR=\"#CCCCCC\"><TD WIDTH=\"50%\" VALIGN=TOP>\n")
      +("<B>input:</B>\n")
      +("<P>\n")
      +("training: <INPUT TYPE=\"FILE\" NAME=\"infile_train\">\n")
      +("<P>\n")
      +("evaluation: <INPUT TYPE=\"FILE\" NAME=\"infile_eval\">\n")
      +("<P>\n")
      +("</TD>\n")
      +("<TD VALIGN=TOP>\n")
      +("<B>options:</B>\n")
      +("<HR>\n")
      +("<B>output:</B>\n")
      +("<P>\n")
      +("<HR>\n")
      +("<B>misc:</B><BR>\n")
      +("<INPUT TYPE=CHECKBOX NAME=\"verbose\" VALUE=\"CHECKED\" "+params.getVal("verbose")+">verbose<BR>\n")
      +("</TD></TR></TABLE>\n")
      +("<P>\n")
      +("<CENTER>\n")
      +("<BUTTON TYPE=BUTTON onClick=\"go_forest(this.form)\">\n")
      +("<B>Go "+APPNAME+"</B></BUTTON>\n")
      +("</CENTER>\n")
      +("</FORM>\n");
    return htm;
  }
  /////////////////////////////////////////////////////////////////////////////
  private static void Learner(MultipartRequest mrequest,HttpServletResponse response)
      throws IOException
  {
    int n_out=0;
    try
    {
      String outtxt = forest_test.RandomForest_test(fileT,fileE);
      outputs.add("<PRE>"+outtxt+"</PRE>");
    }
    catch (Exception e)
    {
      errors.add(""+e);
    }

    PrintWriter out_log=new PrintWriter(
      new BufferedWriter(new FileWriter(logfile,true)));
    out_log.printf("%s\t%s\t%d\n",datestr,REMOTEHOST,n_out); 
    out_log.close();
  }
  /////////////////////////////////////////////////////////////////////////////
  private static String JavaScript() throws IOException
  {
    return(
"function go_init(form)"+
"{\n"+
"  var i;\n"+
"  form.verbose.checked=false;\n"+
"}\n"+
"function checkform(form)\n"+
"{\n"+
"  if (!form.infile_train.value \n"+
"     && !form.infile_eval.value \n"+
"     ) {\n"+
"    alert('ERROR: Incomplete input.');\n"+
"    return 0;\n"+
"  }\n"+
"  return 1;\n"+
"}\n"+
"function go_forest(form)\n"+
"{\n"+
"  if (!checkform(form)) return;\n"+
"  form.forest.value='TRUE';\n"+
"  form.submit()\n"+
"}\n"
    );
  }
  /////////////////////////////////////////////////////////////////////////////
  private static String HelpHtm()
  {
    String htm=
    ("<B>"+APPNAME+" Help</B><P>\n"+
    "<P>\n"+
    "Both the training and evaluation, a.k.a. validation input files must be \n"+
    "a set of molecules in a format such as SMILES or MDL molfile.  The names of the\n"+
    "molecules should indicate one of two classifications, e.g. \"active\" vs.\n"+
    "\"inactive\", or \"toxic\" vs. \"nontoxic\".  Integers (0 or 1) are also fine.\n"+
    "<P>\n"+
    "This web app will generate a set of molecular descriptors using the ChemAxon\n"+
    "JChem API, and submit these data to Weka for random forest analysis. \n"+
    "<P>\n"+
    "Configured with limit N_MAX = "+N_MAX+"\n"+
    "<P>\n"+
    "Thanks to Weka and ChemAxon for use of their excellent software in this application.\n"+
    "<P>\n"+
    "author: Jeremy Yang\n");
    return htm;
  }
  /////////////////////////////////////////////////////////////////////////////
  public void init(ServletConfig conf) throws ServletException
  {
    super.init(conf);
    CONTEXT=getServletContext();	// inherited method
    CONFIG=conf;
    // read servlet parameters (from web.xml):
    try { APPNAME=conf.getInitParameter("APPNAME"); }
    catch (Exception e) { APPNAME=this.getServletName(); }
    UPLOADDIR=conf.getInitParameter("UPLOADDIR");
    if (UPLOADDIR==null)
      throw new ServletException("Please supply UPLOADDIR parameter");
    SCRATCHDIR=conf.getInitParameter("SCRATCHDIR");
    if (SCRATCHDIR==null) SCRATCHDIR="/tmp";
    LOGDIR=conf.getInitParameter("LOGDIR")+CONTEXTPATH;
    if (LOGDIR==null) LOGDIR="/usr/local/tomcat/logs"+CONTEXTPATH;
    try { N_MAX=Integer.parseInt(conf.getInitParameter("N_MAX")); }
    catch (Exception e) { N_MAX=1000; }
  }
  /////////////////////////////////////////////////////////////////////////////
  public void doGet(HttpServletRequest request,HttpServletResponse response)
      throws IOException,ServletException
  {
    doPost(request,response);
  }
}
