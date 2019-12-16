<?php
$f = fopen ("data.txt", "r");
$ln= 0;
echo "<table border='1' style='width: 100%;'>";
echo "<tr><th>Data</th><tr>";
while ($line= fgets ($f)) {
    echo "<tr>";
    ++$ln;
    $array = explode("    ",$line);
    foreach($array as $value){
        echo "<td>".$value."</td>";
    }
    echo "</tr>";
}
fclose ($f);
?>