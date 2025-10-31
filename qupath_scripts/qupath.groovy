// QuPath Groovy script
// Import bbox & point prompts from JSON and run SAM per prompt.
// Works without JsonSlurper; uses GsonTools and JavaFX Platform.runLater to set the image.

import static qupath.lib.gui.scripting.QPEx.*
import java.nio.file.Paths
import qupath.lib.io.PathIO

import javafx.application.Platform
import qupath.lib.io.GsonTools
import qupath.lib.objects.PathObject
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane

import java.util.concurrent.CountDownLatch

// ========== USER SETTINGS ==========
def DATA_DIR     = "/home/carlos/Pictures/samj_rebuttal/cellpose/"
def QUPATH_DIR   = DATA_DIR + "/qupath"
def SAVE_DIR   = DATA_DIR + "/save"

def dir = new File(SAVE_DIR)
if (!dir.exists()) {
    dir.mkdirs()
}

def SERVER_URL   = "http://localhost:8000/sam/"
def VERIFY_SSL   = false
def MODEL        = org.elephant.sam.entities.SAMType.VIT_T
def OUTPUT_TYPE  = org.elephant.sam.entities.SAMOutput.MULTI_SMALLEST
def SET_NAME     = true
def RANDOM_COLOR = true
// ===================================

// ---------- Helpers ----------
List<PathObject> makeBBoxPrompts(double[][] bboxList, ImagePlane plane, String namePrefix = "BBox") {
    if (bboxList == null || bboxList.length == 0) return Collections.emptyList()
    def out = new ArrayList<PathObject>(bboxList.length)
    int i = 0
    for (double[] bb : bboxList) {
        if (bb == null || bb.length < 4) continue
        def roi = ROIs.createRectangleROI(bb[0], bb[1], bb[2], bb[3], plane)
        def obj = PathObjects.createAnnotationObject(roi, null)
        obj.setName("${namePrefix}_${++i}")
        out.add(obj)
    }
    return out
}

// One PathObject per group: [[[x,y],[x,y],...], ...]
List<PathObject> makePointGroupPrompts(double[][][] groups, ImagePlane plane, String namePrefix = "PointSet") {
    if (groups == null || groups.length == 0) return Collections.emptyList()
    def out = new ArrayList<PathObject>(groups.length)
    int i = 0
    for (double[][] grp : groups) {
        if (grp == null || grp.length == 0) continue
        double[] xs = new double[grp.length]
        double[] ys = new double[grp.length]
        for (int k = 0; k < grp.length; k++) {
            if (grp[k] == null || grp[k].length < 2) continue
            xs[k] = grp[k][0]; ys[k] = grp[k][1]
        }
        def roi = ROIs.createPointsROI(xs, ys, plane)
        def obj = PathObjects.createAnnotationObject(roi, null)
        obj.setName("${namePrefix}_${++i}")
        out.add(obj)
    }
    return out
}

// Flat list -> ONE multi-point prompt: [[x,y], ...]
List<PathObject> makeFlatPointPrompt(double[][] pts, ImagePlane plane, String name = "Points_1") {
    if (pts == null || pts.length == 0) return Collections.emptyList()
    double[] xs = new double[pts.length]
    double[] ys = new double[pts.length]
    for (int i = 0; i < pts.length; i++) {
        if (pts[i] == null || pts[i].length < 2) continue
        xs[i] = pts[i][0]; ys[i] = pts[i][1]
    }
    def roi = ROIs.createPointsROI(xs, ys, plane)
    def obj = PathObjects.createAnnotationObject(roi, null)
    obj.setName(name)
    return [obj]
}

// Run one SAM task for a set of prompts (foreground/background lists)
void runSamForPrompts(List<PathObject> fg, List<PathObject> bg = Collections.emptyList()) {
    def viewer = getCurrentViewer()
    def task = org.elephant.sam.tasks.SAMDetectionTask.builder(viewer)
        .server(org.elephant.sam.Utils.createRenderedServer(viewer))
        .serverURL("http://127.0.0.1:8000/sam/")
        .verifySSL(false)
        .model(org.elephant.sam.entities.SAMType.VIT_T)
        .outputType(org.elephant.sam.entities.SAMOutput.MULTI_SMALLEST)
        .setName(true)
        .setRandomColor(true)
        .addForegroundPrompts(fg)
        .addBackgroundPrompts(bg)
        .build()

    task.setOnSucceeded(event -> {
        List<PathObject> detected = task.getValue()
        detected.get(0).setName("BLABLA")
        if (detected == null || detected.isEmpty()) {
            print "No objects detected"
            return
        }
        Platform.runLater(() -> {
            def h = getCurrentHierarchy()
            h.addObjects(detected)
            h.getSelectionModel().clearSelection()
            h.fireHierarchyChangedEvent(this)
        })
    })
    Platform.runLater(task)
}

// ---------- Main ----------
def gson = GsonTools.getInstance(true)

def project = getProject()
if (project == null) {
    print "No project open."
    return
}

// Sort a mutable copy of the image list
def imageList = new ArrayList(project.getImageList())
imageList.sort { a, b -> String.CASE_INSENSITIVE_ORDER.compare(a.getImageName() ?: "", b.getImageName() ?: "") }

def plane = ImagePlane.getPlane(0, 0)

for (entry in imageList) {
    def name = entry.getImageName()
    if (!(name?.endsWith("_img.png")))
        continue

    print "\n🖼️ Processing: ${name}"

    // --- Load this entry into the viewer on the UI thread ---
    def imageData = entry.readImageData()
    def latch = new CountDownLatch(1)
    Platform.runLater({
        getCurrentViewer().setImageData(imageData)
        latch.countDown()
    })
    latch.await()  // wait until the viewer has the image

    def bboxJson  = new File("${QUPATH_DIR}/bbox_prompts_${name}.json")
    def pointJson = new File("${QUPATH_DIR}/point_prompts_${name}.json")

    if (!bboxJson.exists() && !pointJson.exists()) {
        print "⚠️ No JSONs found for ${name}"
        continue
    }

    // ---- BBoxes: run one SAM task per bbox prompt ----
    if (bboxJson.exists()) {
        try {
            double[][] bboxList = gson.fromJson(bboxJson.text, double[][].class).take(2)
            def boxPrompts = makeBBoxPrompts(bboxList, plane, "BBox")
            print "📦 BBoxes: ${boxPrompts.size()} prompts from ${bboxJson.name}"
            for (PathObject p : boxPrompts) {
                runSamForPrompts([p])  // foreground-only per bbox
            }
            def path = Paths.get(SAVE_DIR + "/" + name + ".geojson").toString()
            print(path)
            def annotations = getAnnotationObjects()
            exportObjectsToGeoJson(annotations, path, "FEATURE_COLLECTION")
        } catch (Exception e) {
            print "❌ Error reading bboxes for ${name}: ${e.message}"
        }
    }
    break

    // ---- Points: grouped (3D) -> per group; flat (2D) -> single prompt ----
    if (pointJson.exists()) {
        try {
            double[][][] groups = gson.fromJson(pointJson.text, double[][][].class)
            if (groups != null) {
                def pointPrompts = makePointGroupPrompts(groups, plane, "PointSet")
                print "📍 Point groups: ${pointPrompts.size()} prompts from ${pointJson.name}"
                for (PathObject p : pointPrompts) {
                    runSamForPrompts([p])
                }
            } else {
                double[][] pts = gson.fromJson(pointJson.text, double[][].class)
                def flatPrompt = makeFlatPointPrompt(pts, plane, "Points_1")
                print "📍 Flat points: ${flatPrompt.size()} prompt from ${pointJson.name}"
                if (!flatPrompt.isEmpty())
                    runSamForPrompts(flatPrompt)
            }
        } catch (Exception e) {
            print "❌ Error reading points for ${name}: ${e.message}"
        }
    }
}

print "\n🎉 Done."
